# import MySQLdb
#
# class operatemysql:
#     #获取链接
#     def get_com(self):
#         try:
#             self.con = MySQLdb.connect(
#                 host='localhost',
#                 port=3306,
#                 user="root",
#                 password="123",
#                 db="news",
#                 charset='utf8')
#             cursor = self.con.cursor()
#         except MySQLdb.Error as e:
#             print("Error %d: %s" % (e.args[0],e.args[1]))
#         return self.con
#
# def main():
#
# if "__name__" == "main":
#     main()
import MySQLdb

conn = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="",
    db="school",
    port=3306,
    charset='utf8'
)
cursor = conn.cursor()
cursor.execute('SELECT * FROM `news`where `id`=13;')
rest = cursor.fetchone()
print(rest)