from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Start session
spark = SparkSession.builder.appName("UKTrafficClustering").getOrCreate()

# Load dataset
df = spark.read.csv("UK_Traffic_Data_Clean.csv", header=True, inferSchema=True)

# Rename required columns
df = df.withColumnRenamed("All Motor Vehicles", "all_motor_vehicles") \
       .withColumnRenamed("Pedal Cycles", "pedal_cycles") \
       .withColumnRenamed("Cars and Taxis", "cars_and_taxis") \
       .withColumnRenamed("LGVs", "lgvs") \
       .withColumnRenamed("All HGVs", "all_HGVs")

# Select only numerical features for clustering
numeric_features = ['all_motor_vehicles', 'pedal_cycles', 'cars_and_taxis', 'lgvs', 'all_HGVs']
df_clean = df.select(numeric_features)

# Handle missing values (if any)
df_clean = df_clean.na.fill(0)

# Assemble features
assembler = VectorAssembler(inputCols=numeric_features, outputCol="features_unscaled")
df_vector = assembler.transform(df_clean)

# Scale features
scaler = MinMaxScaler(inputCol="features_unscaled", outputCol="features_scaled")
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# Apply KMeans clustering
kmeans = KMeans(featuresCol="features_scaled", k=4, seed=1)
model = kmeans.fit(df_scaled)

# Predict cluster labels
df_clustered = model.transform(df_scaled)

# Evaluate using Silhouette score
evaluator = ClusteringEvaluator(featuresCol="features_scaled", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(df_clustered)

print("\n✅ Silhouette Score:", silhouette)

# Show sample cluster results
df_clustered.select(numeric_features + ["prediction"]).show(10)

spark.stop()
