use baseball;

DROP TABLE IF EXISTS t_rolling_lookup;

CREATE TABLE t_rolling_lookup AS SELECT g.game_id, local_date, batter, atBat, Hit
    FROM batter_counts bc
    JOIN game g ON g.game_id = bc.game_id;
