view_list = []
time_list = []
all_list = []
f=open("/data1/liufengyi/all_datasets/list_nerft/train.txt","r")
for line in f:
        num = int(line.strip('\n').split(',')[0])
        view = num//100
        time = num%100
        all_list.append((view,time))
        view_list.append(view)
        time_list.append(time)

# from xlsxwriter.workbook import Workbook
# import xlwt
# import xlrd
# from xlutils.copy import copy
# workbook = Workbook(r'test1.xlsx') # 创建xlsx
# worksheet = workbook.add_worksheet('A') # 添加sheet
# red = workbook.add_format({'color':'red'}) # 颜色对象
# styleBlueBkg = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
# for i in range(0,9):
#         for j in range(0,100):
#                 if (i,j) in all_list:
#                         print('liu')
#                         ws.write(i,col,ro.cell(i, col).value,styleBlueBkg)

#                         # worksheet.write_rich_string(i, j, "ok")
#                         worksheet.write(j,i,'train')
# # worksheet.write(0, 0, 'sentences') # 0，0表示row，column，sentences表示要写入的字符串
# # test_list = ["我爱", "中国", "天安门"]
# # test_list.insert(1, red) # 将颜色对象放入需要设置颜色的词语前面
# # print(test_list)
# # worksheet.write_rich_string(1, 0, *test_list) # 写入工作簿
# workbook.close() # 记得关闭
import xlwt
import xlrd
from xlutils.copy import copy
#创建execl
def create_execl(file_name):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Info')
    for i in range(0,9):
        for j in range(0,100):
            if (i,j) in all_list:
                print('liu')
                ws.write(j,i,"train")
    # ws.write(0, 0, "1")
    # ws.write(1, 0, "2")
    # ws.write(2, 0, "3")
    # ws.write(3, 0, "2")
    wb.save(file_name)
#单元格上色
def color_execl(file_name):
    styleBlueBkg1 = xlwt.easyxf('pattern: pattern solid, fore_colour blue;')  # 红色
    styleBlueBkg = xlwt.easyxf('pattern: pattern solid, fore_colour red;')  # 红色
    rb = xlrd.open_workbook(file_name)      #打开t.xls文件
    ro = rb.sheets()[0]                     #读取表单0
    wb = copy(rb)                           #利用xlutils.copy下的copy函数复制
    ws = wb.get_sheet(0)                    #获取表单0
    col = 0                                 #指定修改的列
    for i in range(ro.nrows):               #循环所有的行
        for j in range(9):

            result = ro.cell(i, j).value
            if result == 'train':                     #判断是否等于2
                ws.write(i,j,ro.cell(i, j).value,styleBlueBkg)
            else:
                ws.write(i,j,ro.cell(i, j).value,styleBlueBkg1)
    wb.save(file_name)
 
# if __name__ == '__main__':
#     file_name = 'test1.xls'
#     wb = xlwt.Workbook()
#     ws = wb.add_sheet('Info')
#     styleBlueBkg = xlwt.easyxf('pattern: pattern solid, fore_colour red;')  # 红色
#     for i in range(0,9):
#         for j in range(0,100):
#             if (i,j) in all_list:
#                 print('liu')
#                 ws.write(j, i, "train",styleBlueBkg)
                
#     wb.save(file_name)

if __name__ == '__main__':
    file_name = '/data1/liufengyi/nerf_t-pl/nerf_t-pl/look.xls'
    create_execl(file_name)
    color_execl(file_name)