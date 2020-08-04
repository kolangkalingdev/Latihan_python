
import mysql.connector as mysql

db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "",
    database = "mysql"
)

cursor = db.cursor()

## defining the Query

query = "SELECT * FROM help_topic;"

## getting records from the table
cursor.execute(query)

## fetching all records from the 'cursor' object
records = cursor.fetchall()

## Showing the data
for record in records:
    print(record)