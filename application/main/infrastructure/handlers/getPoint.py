import cv2


def get_point(event, x, y, flags, param):
    # 鼠标单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出坐标
        print('坐标值: ', x, y)
        # 在传入参数图像上画出该点
        # cv2.circle(param, (x, y), 1, (255, 255, 255), thickness=-1)
        img = param.copy()
        # 输出坐标点的像素值
        print('像素值：', param[y][x])  # 注意此处反转，(纵，横，通道)
        # 显示坐标与像素
        text = "(" + str(x) + ',' + str(y) + ')' + str(param[y][x])
        cv2.putText(img, text, (0, param.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # 定义两幅图像
    image = cv2.imread(r'E:\repos\Vehicle-tracking\videos\sample_Moment1.jpg')
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # 定义两个窗口 并绑定事件 传入各自对应的参数
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_point, image)

    # 显示图像
    while (True):
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
