from flask import *
import pickle
from flask_session import Session
import numpy as np
from flask_mysqldb import MySQL
import csv
modelFile = open('testv4.pkl','rb')
model = pickle.load(modelFile)
 
app=Flask(__name__)
app.secret_key = 'xse'

# app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)
# mysql things
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'hack ni'
 
mysql = MySQL(app)


#Creating a connection cursor
# cursor = mysql.connection.cursor()
 
#Executing SQL Statements
# cursor.execute(''' CREATE TABLE table_name(field1, field2...) ''')
# cursor.execute(''' INSERT INTO table_name VALUES(v1,v2...) ''')
# cursor.execute(''' DELETE FROM table_name WHERE condition ''')
 
#Saving the Actions performed on the DB
# mysql.connection.commit()
 
#Closing the cursor
# cursor.close()

#--------------------------------------------------
# routings,endpoints
@app.route('/')
def home():
    return render_template('landingpage.html')




@app.route('/user/<jesan>')
def user(name):
    file = open()
    
    return render_template('deshbord.html')

@app.route('/user/edit/<name>')
def userEdit(name):
    return render_template('add_amp.html')



@app.route('/login/<t>',methods=['POST',"GET"])
def login(t):
        if request.method == 'POST':
            uname=request.form['uname']  
            passwrd=request.form['pass']  
            if uname=="ayush" and passwrd=="google":  
                return "Welcome %s" %uname  
            else:
                return 'Wrong password!'
        if t == 'r':
            return render_template('lor.html',tt=t)
        else:
            return render_template('lor.html',tt=t)


@app.route('/register',methods=['POST'])
def register():
    name = request.form['username']
    email = request.form['email']
    org = request.form['org']
    passs = request.form['passs']
    try:
        cur = mysql.connect.cursor()
        cur.execute("INSERT INTO `hrs`(organization, name, pass, email) VALUES (%s,%s,%s,%s)",(org,name,passs,email))
    except Exception as e:
       return(str(e))
    return 'register'

@app.route('/contactus',methods=['POST','GET'])
def contactus():
     if request.method == 'POST':
          return redirect('/')
     return render_template('contact.html')
# @app.route('/login',methods=)



if __name__=="__main__":
    app.run(debug=True)