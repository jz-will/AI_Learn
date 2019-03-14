import MySQLdb

class MysqlSearch(object):
    #初始化
    def __init__(self):
        self.get_conn()
    #获取链接
    def get_conn(self):
        try:
            self.conn = MySQLdb.connect(
                host='localhost',
                user="root",
                password="",
                db="school",
                port=3306,
                charset='utf8'
            )
        except MySQLdb.Error as e:
            print("Error %d: %s" % (e.args[0],e.args[1]))
    #关闭链接
    def close_conn(self):
        try:
            if self.conn:
                self.conn.close()
        except MySQLdb.Error as e:
            print("Error:" %e)
    #获取一条数据
    def get_one(self):
        sql = 'select * from `news` where `id`= 13;'    #%d to avoid
        cursor = self.conn.cursor()
        cursor.execute(sql,)
        rest = cursor.fetchone()
        print(rest)
        cursor.close()
        self.close_conn()

def main():
    obj = MysqlSearch()
    obj.get_one()

if __name__ == "__main__":
    main()