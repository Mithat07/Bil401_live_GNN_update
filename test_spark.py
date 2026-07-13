from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('test').getOrCreate()
spark.range(5).write.mode('overwrite').text('runs/kafka/test_output')
print('OK')
