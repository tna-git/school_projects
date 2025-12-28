import pyspark
from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType


############################# PART 1 #############################

# IDEA: Create df, then use unique or a for loop to find anomaly number
spark = SparkSession.builder.master("local[*]").appName("ProcessDatFile").getOrCreate()

schema = StructType([StructField("val", DoubleType(), True)])
# # Read as plain text first
file_path = "log.dat"
rdd = spark.sparkContext.textFile(file_path)
rdd_split = rdd.map(lambda l: l.strip()).filter(lambda l: l != "").map(lambda l: (float(l),))
schema = StructType([StructField("val", DoubleType(), True)])

df = spark.createDataFrame(rdd_split, schema=schema)
df.show(5)
print(df.unique().show())
df.unique_val = df.distinct().collect()
print(len(df.unique_val))
# FOUND ANOMOALY NUMBER: 678304493.3894111




######### PART 2############

# IDEA: Loop through old files and use in command to see if anomoaly number is in the file.
import os
 for filename