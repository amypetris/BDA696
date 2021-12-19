sleep 20
if ! mysql -h mariadb -uroot -ppassword -P3306 --protocol=TCP -e "USE baseball"; then
  mysql -h mariadb -uroot -ppassword  -P3306 --protocol=TCP -e "CREATE DATABASE baseball;"
  mysql -h mariadb -uroot -ppassword   -P3306 --protocol=TCP baseball < baseball.sql
  mysql -h mariadb -uroot -ppassword -P3306 --protocol=TCP baseball < final.sql
fi
mysql -h mariadb -uroot -ppassword -P3306 --protocol=TCP baseball < final.sql

python3 final.py
