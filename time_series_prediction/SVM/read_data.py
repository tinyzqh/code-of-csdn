import xlrd
import matplotlib.pyplot as plt
def read_20180829():
    fname = "20180829.xlsx"
    bk = xlrd.open_workbook(fname)
    # shxrange = range(bk.nsheets)
    try:
        sh = bk.sheet_by_name("Sheet1")
    except:
        print("no sheet in %s named Sheet1" % fname)
    # 获取行数
    nrows = sh.nrows
    # 获取列数
    ncols = sh.ncols
    # print("nrows %d, ncols %d" % (nrows, ncols))
    # 获取第一行第一列数据
    cell_value = sh.cell_value(1, 0)
    # print(cell_value)
    time = []
    single1 = []
    single2 = []
    single3 = []
    # 获取各行数据
    for i in range(1, nrows):
        row_data = sh.cell_value(i, 0)
        # print('time', row_data)
        time.append(row_data)
    for i in range(1, nrows):
        row_data = sh.cell_value(i, 1)
        # print('a', row_data)
        single1.append(row_data)
    for i in range(1, nrows):
        row_data = sh.cell_value(i, 2)
        # print('a', row_data)
        single2.append(row_data)
    for i in range(1, nrows):
        row_data = sh.cell_value(i, 3)
        # print('a', row_data)
        single3.append(row_data)
    return time,single1,single2,single3
# time,single1,single2,single3 = read_20180829()
# plt.subplot(2, 2, 1)
# plt.plot(time, single1)
# plt.xlabel("time")
# plt.ylabel("single1")
# plt.subplot(2, 2, 2)
# plt.plot(time, single2)
# plt.xlabel("time")
# plt.ylabel("single2")
# plt.subplot(2, 2, 3)
# plt.plot(time, single3)
# plt.xlabel("time")
# plt.ylabel("single3")
# plt.show()