import sys

from pyspark.sql import SparkSession

# Using MYSQL


def main():
    spark = (
        SparkSession.builder.config(
            "spark.jars",
            ".venv/lib/python3.8/site-packages/mysql-connector-java-8.0.26/mysql-connector-java-8.0.26.jar",
        )
        .master("local")
        .appName("Homework 3")
        .getOrCreate()
    )

    # Batter Average SQL code graciously donated by @dafrenchyman
    rolling_lookup = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball")
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .option(
            "dbtable",
            """
            (SELECT g.game_id, local_date, batter, atBat, Hit
            FROM batter_counts bc
            JOIN game g ON g.game_id = bc.game_id
            WHERE atBat > 0) t_rolling_lookup
            """,
        )
        .option("user", "root")
        .option("password", "420Amy!!")
        .load()
    )

    rolling_lookup.show()

    rolling_lookup.createOrReplaceTempView("t_rolling_lookup")

    rolling = spark.sql(
        """
        SELECT
        rl1.batter, rl1.game_id, rl1.local_date, SUM(rl2.Hit) / SUM(rl2.atBat) AS BA
        FROM t_rolling_lookup rl1
        JOIN t_rolling_lookup rl2 ON rl1.batter = rl2.batter
        AND rl2.local_date BETWEEN DATE_SUB(rl1.local_date, 100) AND rl1.local_date
        GROUP BY rl1.batter, rl1.game_id, rl1.local_date
        """
    )

    rolling.show()


if __name__ == "__main__":
    sys.exit(main())
