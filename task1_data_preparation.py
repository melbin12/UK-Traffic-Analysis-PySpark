from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# 1. Create Spark session
spark = SparkSession.builder \
    .appName("UKTrafficDataPreparation") \
    .getOrCreate()

# 2. Load CSV file
df = spark.read.csv("UK_Traffic_Data_Clean.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)

# 3. Drop unwanted columns (e.g., location data if not needed)
columns_to_drop = ['Direction of Travel', 'Easting', 'Northing']
df_clean = df.drop(*columns_to_drop)

# 🔁 4. ✅ RENAME COLUMNS WITH SPACES — ADD HERE
df_clean = df_clean.withColumnRenamed("All Motor Vehicles", "all_motor_vehicles") \
                   .withColumnRenamed("Pedal Cycles", "pedal_cycles") \
                   .withColumnRenamed("Cars and Taxis", "cars_and_taxis") \
                   .withColumnRenamed("Buses and Coaches", "buses_and_coaches") \
                   .withColumnRenamed("Two-Wheeled Motor Vehicles", "two_wheeled_motor_vehicles") \
                   .withColumnRenamed("All HGVs", "all_HGVs") \
                   .withColumnRenamed("LGVs", "lgvs")

# 5. Now handle missing values
numeric_cols = ['all_motor_vehicles', 'pedal_cycles', 'cars_and_taxis']
for col_name in numeric_cols:
    avg_value = df_clean.select(mean(col_name)).collect()[0][0]
    df_clean = df_clean.fillna({col_name: avg_value})

# 6. Outlier removal using IQR method
q1 = df_clean.approxQuantile("all_motor_vehicles", [0.25], 0.05)[0]
q3 = df_clean.approxQuantile("all_motor_vehicles", [0.75], 0.05)[0]
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df_no_outliers = df_clean.filter((col("all_motor_vehicles") >= lower) & (col("all_motor_vehicles") <= upper))

# 7. Feature scaling using MinMaxScaler
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_unscaled")
df_vector = assembler.transform(df_no_outliers)

scaler = MinMaxScaler(inputCol="features_unscaled", outputCol="features_scaled")
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# 8. Show final result
df_scaled.select("features_scaled").show(5)

# 9. Stop Spark session
spark.stop()
