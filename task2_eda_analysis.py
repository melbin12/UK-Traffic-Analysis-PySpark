from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, col

# Start Spark Session
spark = SparkSession.builder.appName("TrafficEDA").getOrCreate()

# Load data
df = spark.read.csv("UK_Traffic_Data_Clean.csv", header=True, inferSchema=True)

# Rename relevant columns
df = df.withColumnRenamed("All Motor Vehicles", "all_motor_vehicles") \
       .withColumnRenamed("Region Name", "region_name") \
       .withColumnRenamed("Road Type", "road_type") \
       .withColumnRenamed("Year", "year") \
       .withColumnRenamed("Pedal Cycles", "pedal_cycles") \
       .withColumnRenamed("Cars and Taxis", "cars_and_taxis") \
       .withColumnRenamed("Buses and Coaches", "buses_and_coaches") \
       .withColumnRenamed("Two-Wheeled Motor Vehicles", "two_wheeled_motor_vehicles") \
       .withColumnRenamed("LGVs", "lgvs") \
       .withColumnRenamed("All HGVs", "all_HGVs")

# 1. Total traffic by year
print("\n🔹 Total Traffic by Year:")
df.groupBy("year").agg(sum("all_motor_vehicles").alias("total_traffic")).orderBy("year").show()

# 2. Region-wise total traffic
print("\n🔹 Region-wise Total Traffic:")
df.groupBy("region_name").agg(sum("all_motor_vehicles").alias("regional_traffic")).orderBy(col("regional_traffic").desc()).show()

# 3. Average vehicle type counts
print("\n🔹 Vehicle Type Statistics:")
df.select("pedal_cycles", "cars_and_taxis", "lgvs", "all_HGVs", "buses_and_coaches").describe().show()

# 4. Traffic by road type
print("\n🔹 Traffic by Road Type:")
df.groupBy("road_type").agg(sum("all_motor_vehicles").alias("traffic_by_road_type")).show()

spark.stop()
