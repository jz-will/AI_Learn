# -*- coding: utf-8 -*-
import MySQLdb


class MysqlSearch(object):
    # 初始化
    def __init__(self):
        self.get_conn()

    # 获取链接
    def get_conn(self):
        try:
            self.conn = MySQLdb.connect(
                host='127.0.0.1',
                user="root",
                password="",
                db="school",
                port=3306,
                charset='utf8'
            )
        except MySQLdb.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))

    # 关闭链接
    def close_conn(self):
        try:
            if self.conn:
                self.conn.close()
        except MySQLdb.Error as e:
            print("Error:" % e)

    # 获取一条数据
    def get_one(self):
        sql = 'select * from `news` where `id`= %d;' % (16)  # %d 避免sql注入
        cursor = self.conn.cursor()
        cursor.execute(sql)
        rest = cursor.fetchone()
        print(rest)
        cursor.close()
        self.close_conn()

    def insertSql(self):
        val = (16, '标题6', '属性6', '图6', '作者6', 234, 67)
        # sql语句中的%s要带引号
        sql = "insert into `news` value (%d, '%s', '%s', '%s', '%s', %d, %d)" % val
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql)
            self.conn.commit()

        except:
            self.conn.rollback()

        cursor.close()
        # self.close_conn()


def main():
    obj = MysqlSearch()
    # obj.insertSql()
    obj.get_one()


if __name__ == "__main__":
    main()
