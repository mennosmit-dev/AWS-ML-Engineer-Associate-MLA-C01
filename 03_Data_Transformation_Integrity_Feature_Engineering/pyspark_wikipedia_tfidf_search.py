# On EMR, in a notebook of PySpark on AWS, backed by EMR cluster, excersises project to for a word find relevant wikipedia pages using TD-IDF (cleaned wiki dataset).


# pyspark_wikipedia_tfidf_search.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, ArrayType, StringType, FloatType
)
from pyspark.sql.functions import udf

from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# Load subset of Wikipedia TSV from S3

S3_PATH = "s3://aws-emr-studio-159107795666-us-east-1/1733157099735/subset-small.tsv"

rawdata = (
    spark.read
         .options(sep="\t")
         .csv(S3_PATH)
)

# Show raw data
rawdata.show(20, truncate=True)


# Add column names
articles = rawdata.toDF("ID", "Title", "Time", "Document")
articles.show(20, truncate=True)

# Remove null documents (TF/IDF cannot handle null text)

null_docs = articles.filter(col("Document").isNull()).count()
print(f"Null documents: {null_docs}")

cleanedArticles = articles.filter(col("Document").isNotNull())
print(f"Null documents after cleaning: {cleanedArticles.filter(col('Document').isNull()).count()}")


# Tokenize documents into words

tokenizer = Tokenizer(inputCol="Document", outputCol="words")
wordsData = tokenizer.transform(cleanedArticles)

# HashingTF: convert tokens -> term frequency vector

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")  # default numFeatures=262144
featurizedData = hashingTF.transform(wordsData)

featurizedData.show(5, truncate=True)


# IDF: scale TF -> TF/IDF

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.show(5, truncate=True)


# Get the hashed feature index for the search term "Swimming"
schema = StructType([StructField("words", ArrayType(StringType()), True)])
term_df = spark.createDataFrame([[["Swimming"]]], schema=schema)

term_hashed = hashingTF.transform(term_df)
feature_row = term_hashed.select("rawFeatures").collect()[0]
wordID = int(feature_row.rawFeatures.indices[0])

print(f'Hashed feature index for word: {wordID}')


# Extract the TF/IDF value at that index for each document as "score"
termExtractor = udf(lambda v: float(v[wordID]), FloatType())

wordDF = rescaledData.withColumn("score", termExtractor(col("features")))
wordDF.select("ID", "Title", "score").show(20, truncate=True)

# Sort by score (descending) to find most relevant docs

sortedResults = (
    wordDF
    .filter(col("score") > 0)
    .orderBy(col("score").desc())
    .select("ID", "Title", "Document", "score")
)

sortedResults.show(20, truncate=100)
