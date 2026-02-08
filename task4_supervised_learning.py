from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Start Spark session
spark = SparkSession.builder.appName("TrafficSupervisedLearning").getOrCreate()

# 2. Load dataset again
df = spark.read.csv("UK_Traffic_Data_Clean.csv", header=True, inferSchema=True)

# Rename required columns
df = df.withColumnRenamed("All Motor Vehicles", "all_motor_vehicles") \
       .withColumnRenamed("Pedal Cycles", "pedal_cycles") \
       .withColumnRenamed("Cars and Taxis", "cars_and_taxis") \
       .withColumnRenamed("LGVs", "lgvs") \
       .withColumnRenamed("All HGVs", "all_HGVs")

# Select features
numeric_cols = ['all_motor_vehicles', 'pedal_cycles', 'cars_and_taxis', 'lgvs', 'all_HGVs']
df = df.select(numeric_cols)
df = df.na.fill(0)

# 3. Assemble and scale features
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_unscaled")
df_vector = assembler.transform(df)

scaler = MinMaxScaler(inputCol="features_unscaled", outputCol="features")
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# 4. Import KMeans to generate pseudo-labels (or use clustering output if saved)
from pyspark.ml.clustering import KMeans
kmeans = KMeans(featuresCol="features", k=4)
model = kmeans.fit(df_scaled)
df_labeled = model.transform(df_scaled).withColumnRenamed("prediction", "label")

# 5. Split into training and testing
train, test = df_labeled.randomSplit([0.7, 0.3], seed=123)

# 6. Train classifier
classifier = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=4)
model = classifier.fit(train)

# 7. Predict
predictions = model.transform(test)

# 8. Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# View some predictions
predictions.select("label", "prediction", "features").show(10)

spark.stop()
