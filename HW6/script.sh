
if ! mysql -h mariadb -uroot -ppassword -P3306 --protocol=TCP -e "USE baseball"; then
  echo "test 1"
  mysql -h mariadb -uroot -ppassword  -P3306 --protocol=TCP -e "CREATE DATABASE baseball;"
  mysql  -h mariadb -uroot -ppassword   -P3306 --protocol=TCP baseball < baseball.sql
fi

mysql -h mariadb -u root -ppassword -P3306 --protocol=TCP baseball < rolling_avg.sql
mysql -h mariadb -u root -ppassword -P3306 --protocol=TCP baseball -e 'SELECT rl1.batter, rl1.game_id, rl1.local_date, SUM(rl2.Hit) / SUM(rl2.atBat) AS BA
      FROM t_rolling_lookup rl1
      JOIN t_rolling_lookup rl2 ON rl1.batter = rl2.batter AND rl2.local_date BETWEEN DATE_SUB(rl1.local_date, INTERVAL 100 DAY) AND rl1.local_date
        WHERE rl1.game_id = 12560
      GROUP BY rl1.batter, rl1.game_id, rl1.local_date;'
