import pyspark
from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
import numpy as np

file_path = "space.dat"
#RDD
spark = SparkSession.builder.master("local[*]").appName("ProcessDatFile").getOrCreate()
schema = StructType([
    StructField("Dimension_1", DoubleType(), True),
    StructField("Dimension_2", DoubleType(), True),
    StructField("Dimension_3", DoubleType(), True),
    StructField("frequency_mhz", DoubleType(), True)
])
rdd = spark.sparkContext.textFile(file_path)
#df:
df = spark.read.option("header", "false").option("inferSchema", "true").option("delimiter", ",").csv(file_path)
df = df.toDF("d1", "d2", "d3", "d4", "d5", "d6")


# Prepare features
assembler = VectorAssembler(
    inputCols=["d1", "d2", "d3", "d4", "d5", "d6"],
    outputCol="features"
)
df_features = assembler.transform(df)

# Normalization
scaler = StandardScaler(inputCol="features", outputCol="scaled", withStd=True, withMean=True)
df_scaled = scaler.fit(df_features).transform(df_features)

# Evaluator & looping through cluster numbers
evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
k_values = list(range(2, 9))  # test k = 2 to 8
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(k=k, seed=42, featuresCol="scaled", predictionCol="cluster")
    model = kmeans.fit(df_scaled)
    preds = model.transform(df_scaled)
    
    score = evaluator.evaluate(preds)
    silhouette_scores.append(score)
    print(f"k={k}  silhouette={score:.4f}", flush = True)

# Silhouette plot
plt.figure(figsize=(7,5))
plt.plot(k_values, silhouette_scores, marker='o', color='steelblue')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Mean Silhouette Score")
plt.grid(True, alpha=0.3)
plt.savefig("silhouette plot")
plt.close()

## CLUSTERING

#Kmeans clustering (k = 4)
kmeans = KMeans(k=4, seed=42, featuresCol="features", predictionCol="cluster")
kmeans_model = kmeans.fit(df_features)
clustered_df = kmeans_model.transform(df_features)

print("Cluster centers:", flush = True)
for i, c in enumerate(kmeans_model.clusterCenters()):
    print(f"Cluster {i}: {c}", flush = True)

#PCA on subset of dimensions (1–4)
assembler_pca = VectorAssembler(inputCols=["d1", "d2", "d3", "d4"],outputCol="features_pca")
features_df = assembler_pca.transform(clustered_df)

pca = PCA(k=3, inputCol="features_pca", outputCol="pca_features")
pca_model = pca.fit(features_df)
pca_transformed = pca_model.transform(features_df)
print("Explained variance for set 1:", pca_model.explainedVariance.toArray(), flush = True)
orientation_vector = np.array(pca_model.pc.toArray())[:,0]
print("orientation:", orientation_vector, flush = True)
# Convert to Pandas for plotting
pca_pd = pca_transformed.select("pca_features", "cluster").toPandas()
pca_pd[['PC1','PC2','PC3']] = pd.DataFrame(pca_pd['pca_features'].tolist(), index=pca_pd.index)

# 3D PCA plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_pd['PC1'], pca_pd['PC2'], pca_pd['PC3'], c=pca_pd['cluster'], cmap='tab10', s=4)
plt.colorbar(sc, label='Cluster')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D PCA (dims 1-4) colored by KMeans cluster")
plt.tight_layout()
fig.savefig("pca_1.jpg", dpi=300)
plt.close()

#pca for dimensions 2,3,4,5
assembler_pca = VectorAssembler(inputCols=["d2", "d3", "d4", "d5"],outputCol="features_pca")
features_df = assembler_pca.transform(clustered_df)

pca = PCA(k=3, inputCol="features_pca", outputCol="pca_features")
pca_model = pca.fit(features_df)
pca_transformed = pca_model.transform(features_df)
print("Explained variance for set 2:", pca_model.explainedVariance.toArray(), flush = True)
orientation_vector = np.array(pca_model.pc.toArray())[:,0]
print("orientation:", orientation_vector, flush = True)

pca_pd = pca_transformed.select("pca_features", "cluster").toPandas()
pca_pd[['PC1','PC2','PC3']] = pd.DataFrame(pca_pd['pca_features'].tolist(), index=pca_pd.index)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_pd['PC1'], pca_pd['PC2'], pca_pd['PC3'], c=pca_pd['cluster'], cmap='tab10', s=4)
plt.colorbar(sc, label='Cluster')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D PCA (dims 2-5) colored by KMeans cluster")
plt.tight_layout()
fig.savefig("pca_2.jpg", dpi=300)
plt.close()


#pca for dimensions 3,4,5,6
assembler_pca = VectorAssembler(inputCols=["d3", "d4", "d5", "d6"],outputCol="features_pca")
features_df = assembler_pca.transform(clustered_df)
pca = PCA(k=3, inputCol="features_pca", outputCol="pca_features")
pca_model = pca.fit(features_df)
pca_transformed = pca_model.transform(features_df)
print("Explained variance for set 3:", pca_model.explainedVariance.toArray(), flush = True)
orientation_vector = np.array(pca_model.pc.toArray())[:,0]
print("orientation:", orientation_vector, flush = True)

pca_pd = pca_transformed.select("pca_features", "cluster").toPandas()
pca_pd[['PC1','PC2','PC3']] = pd.DataFrame(pca_pd['pca_features'].tolist(), index=pca_pd.index)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_pd['PC1'], pca_pd['PC2'], pca_pd['PC3'], c=pca_pd['cluster'], cmap='tab10', s=4)
plt.colorbar(sc, label='Cluster')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D PCA (dims 3-6) colored by KMeans cluster")
plt.tight_layout()
fig.savefig("pca_3.jpg", dpi=300)
plt.close()