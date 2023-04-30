import csv
import mysql.connector

# MySQL connection settings
HOST = 'localhost'
USER = 'root'
PASSWORD = ''
DATABASE = 'hack ni'

# Open the CSV file
with open('data2.csv', 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)

    # Connect to the MySQL database
    conn = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
    cursor = conn.cursor()

    # Iterate over each row in the CSV file
    for row in csvreader:
        # Insert the row into the database
        query = ("INSERT INTO table_name ( `organization`, `code`, `Name`, `itter`, `perf`, `Age`, `BusinessTravel`, `DailyRate`, `DistanceFromHome`, `Education`, `EmployeeCount`, `EmplyooNumber`, `EnvironmentSatisfaction`, `Gender`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StandardHours`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTImesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`, `lastChange`, `created`)"
        "VALUES (%s,%s, %s, %s,%s,%s,%s ,%s,%s,%s  ,%s,%s, %s, %s,%s,%s,%s ,%s,%s,%s   ,%s,%s, %s, %s,%s,%s,%s ,%s,%s,%s ,%s,%s,%s,%s,%s)")
        values = ('ibm',row[0], row[36],row[2], row[25],row[1],row[3], row[],row[],row[])
        cursor.execute(query, values)

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()
