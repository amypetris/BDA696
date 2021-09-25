-- USING MYSQL
-- Formatted with https://www.dpriver.com/pp/sqlformat.htm

USE baseball;

-- Create hits table with only hits and no outs
DROP TABLE IF EXISTS only_hits;
CREATE TABLE only_hits AS
  SELECT *
  FROM   hits
  WHERE  type = 'H';

-- Append local date to hits table so i dont have to join with game table every time
DROP TABLE IF EXISTS hits_with_date;
CREATE TABLE hits_with_date AS
  SELECT h.*,
         g.local_date
  FROM   only_hits h
         JOIN game g
           ON h.game_id = g.game_id;

-- append local date to atbats table
DROP TABLE IF EXISTS atbats_with_date;
CREATE TABLE atbats_with_date AS
  SELECT ab.*,
         g.local_date
  FROM   atbat_r ab
         JOIN game g
           ON ab.game_id = g.game_id;

-- create hits by year table
DROP TABLE IF EXISTS hits_by_year;
CREATE TABLE hits_by_year AS
  SELECT Count(hits_id)   AS hits,
         batter,
         Year(local_date) AS year
  FROM   hits_with_date
  GROUP  BY batter,
            year;

-- create atbats by year table
DROP TABLE IF EXISTS atbats_by_year;
CREATE TABLE atbats_by_year AS
  SELECT Count(atbat_r_id) AS atbats,
         batter,
         Year(local_date)  AS year
  FROM   atbats_with_date
  GROUP  BY batter,
            year;

-- create batting averge by year table
DROP TABLE IF EXISTS batting_avg_by_year;
CREATE TABLE batting_avg_by_year AS
  SELECT h.hits / ab.atbats AS batting_average,
         h.hits,
         ab.atbats,
         h.batter,
         h.year
  FROM   hits_by_year h
         JOIN atbats_by_year ab
           ON h.batter = ab.batter
              AND h.year = ab.year;

-- create historic hits utilizing hits_by_year
DROP TABLE IF EXISTS historic_hits;
CREATE TABLE historic_hits AS
  SELECT Sum(hits) AS hits,
         batter
  FROM   hits_by_year
  GROUP  BY batter;

-- create historic atbats using atbats_by_year
DROP TABLE IF EXISTS historic_atbats;
CREATE TABLE historic_atbats AS
  SELECT Sum(atbats) AS atbats,
         batter
  FROM   atbats_by_year
  GROUP  BY batter;

-- create historic batting average table
DROP TABLE IF EXISTS historic_batting_average;
CREATE TABLE historic_batting_average AS
  SELECT hits / atbats AS batting_average,
         h.batter
  FROM   historic_hits h
         JOIN historic_atbats ab
           ON h.batter = ab.batter;

-- create hits by date table used in the rolling ba
DROP TABLE IF EXISTS hits_by_date;
CREATE TABLE hits_by_date AS
  SELECT Count(hits_id) AS hits,
         batter,
         local_date
  FROM   hits_with_date
  GROUP  BY batter,
            local_date;

-- create at bats by date table used in the rolling ba
DROP TABLE IF EXISTS atbats_by_date;
CREATE TABLE atbats_by_date AS
  SELECT Count(atbat_r_id) AS atbats,
         batter,
         local_date
  FROM   atbats_with_date
  GROUP  BY batter,
            local_date;

-- create rolling count of hits
DROP TABLE IF EXISTS hits_rolling;
CREATE TABLE hits_rolling AS
  SELECT Sum(h2.hits) AS hits,
         h.batter,
         h.local_date
  FROM   hits_by_date h
         JOIN hits_by_date h2
           ON h.batter = h2.batter
  WHERE  Datediff(h.local_date, h2.local_date) BETWEEN 0 AND 100
  GROUP  BY h.batter,
            h.local_date;

-- create rolling count of at bats
DROP TABLE IF EXISTS atbats_rolling;
CREATE TABLE atbats_rolling AS
  SELECT Sum(ab2.atbats) AS atbats,
         ab.batter,
         ab.local_date
  FROM   atbats_by_date ab
         JOIN atbats_by_date ab2
           ON ab.batter = ab2.batter
  WHERE  Datediff(ab.local_date, ab2.local_date) BETWEEN 0 AND 100
  GROUP  BY ab.batter,
            ab.local_date;

-- create rolling batting average
DROP TABLE IF EXISTS batting_avg_rolling;
CREATE TABLE batting_avg_rolling AS
  SELECT hits / atbats AS rolling_avg,
         h.batter,
         h.local_date,
         h.hits,
         ab.atbats
  FROM   hits_rolling h
         JOIN atbats_rolling ab
           ON h.batter = ab.batter
              AND h.local_date = ab.local_date;

-- Sanity check
-- Found some anomalies in data
-- cases where batting average is greater than 1 (13 cases)
SELECT *
FROM   batting_avg_rolling
WHERE  hits > atbats;
/* MySQL [baseball]> select * from batting_avg_rolling where hits>atbats;
+-------------+--------+---------------------+------+--------+
| rolling_avg | batter | local_date          | hits | atbats |
+-------------+--------+---------------------+------+--------+
|      1.5000 | 445196 | 2007-03-22 19:05:00 |    3 |      2 |
|      1.3333 | 472528 | 2010-03-09 13:05:00 |    4 |      3 |
|      1.5000 | 458675 | 2008-02-28 13:05:00 |    3 |      2 |
|      3.0000 | 456124 | 2010-03-03 13:05:00 |    3 |      1 |
|      1.5000 | 461235 | 2008-03-01 13:05:00 |    6 |      4 |
|      3.0000 | 502973 | 2008-02-29 13:10:00 |    3 |      1 |
|      3.0000 | 501896 | 2008-02-29 13:10:00 |    3 |      1 |
|      1.5000 | 451259 | 2008-03-01 13:05:00 |    3 |      2 |
|      2.0000 | 461235 | 2008-02-29 19:05:00 |    4 |      2 |
|      2.0000 | 452672 | 2010-03-09 13:05:00 |    2 |      1 |
|      1.3333 | 435459 | 2008-02-29 19:05:00 |    4 |      3 |
|      1.3333 | 407871 | 2008-02-29 19:05:00 |    4 |      3 |
|      2.0000 | 445139 | 2007-03-04 13:05:00 |    2 |      1 |
+-------------+--------+---------------------+------+--------+

*first result there are 2 hits recorded for game 256 however there are no atbats recorded for that game and batter
*second resultthere is a hit recorded for game 9024 however there is not atbats recorded for that game and batter
*going to assume the same for the other 11 cases
*/