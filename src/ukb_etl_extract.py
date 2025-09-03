from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit
from delta import configure_spark_with_delta_pip

import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['SPARK_HOME'] = "/external/rprshnas01/netdata_kcni/dflab/tools/general/distributed-computing/spark/3.5.0/"

def get_params(params_file: str) -> dict:
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params

def start_spark() -> SparkSession:
    logger.info("Starting Spark session...")
    builder = (
        SparkSession.builder.appName('Subsetting UK Biobank Data')
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    logger.info("Spark session started.")
    return spark

def get_df_ukb(spark: SparkSession, path_ukb_delta: str):
    logger.info("Loading UK Biobank main dataset...")
    df_ukb = spark.read.format("delta").load(path_ukb_delta)
    logger.info("UK Biobank main dataset loaded.")
    return df_ukb

def subset_df_ukb(df_ukb, field_list: list, instance_list: list):
    logger.info("Subsetting UK Biobank dataset by fields and instances...")
    selected_cols = ['eid', 'instance'] + field_list
    df_subset = df_ukb.select(*selected_cols).filter(col('instance').isin(instance_list))
    logger.info("Subset created.")
    return df_subset

def exclude_ids_filter(df_subset, spark: SparkSession, path_exclude_ids: str):
    if path_exclude_ids:
        logger.info(f"Loading exclusion list from {path_exclude_ids}...")
        exclude_ids = (
            spark.read.text(path_exclude_ids)
            .withColumnRenamed("value", "eid")
            .distinct()
            .withColumn("eid", col("eid").cast("long"))
        )
        logger.info("Filtering out excluded IDs...")
        df_subset = df_subset.join(exclude_ids, on="eid", how="left_anti")
        logger.info("Excluded IDs removed from dataset.")
    return df_subset

def get_dictionary(spark: SparkSession, path_dictionary: str):
    logger.info("Loading dictionary...")
    df_dictionary = spark.read.csv(path_dictionary, sep='\t', header=True)
    logger.info("Dictionary loaded.")
    return df_dictionary

def create_column_name_mapping_dictionary(df_subset, df_dictionary):
    logger.info("Creating column name mapping dictionary...")
    df_dictionary = df_dictionary.withColumn(
        "NewColumn", 
        concat(df_dictionary['Field'], lit(' (FieldID: '), df_dictionary['FieldID'], lit(')'))
    )
    df_field_name_map = df_dictionary.select('FieldID', 'Field', 'NewColumn')
    df_field_name_map = df_field_name_map.filter(col('FieldID').isin(df_subset.columns))
    mapping = df_field_name_map.toPandas().set_index('FieldID')['NewColumn'].to_dict()
    logger.info("Column name mapping dictionary created.")
    return mapping

def rename_columns(df_subset, mapping: dict):
    logger.info("Renaming columns...")
    df_subset = df_subset.toPandas().rename(columns=mapping)
    logger.info("Columns renamed.")
    return df_subset

def write_csv(df_pandas, path_output: str):
    logger.info(f"Writing CSV to: {path_output}...")
    df_pandas.to_csv(path_output, index=False)
    logger.info("CSV written.")

def extract_cohort(
    params_file: str = 'params.yaml'
):
    params = get_params(params_file)
    main_params = params["ukbiobank"]["main"]

    spark = start_spark()

    df_ukb = get_df_ukb(spark, main_params["delta"])
    df_subset = subset_df_ukb(df_ukb, main_params["fields"], main_params["instances"])

    # Apply exclude_ids filter if present
    exclude_ids_path = main_params.get("exclude_ids")
    df_subset = exclude_ids_filter(df_subset, spark, exclude_ids_path)

    df_dictionary = get_dictionary(spark, main_params["dictionary"])
    mapping = create_column_name_mapping_dictionary(df_subset, df_dictionary)

    df_subset_renamed = rename_columns(df_subset, mapping)
    write_csv(df_subset_renamed, main_params["output"])

if __name__ == "__main__":
    extract_cohort()

