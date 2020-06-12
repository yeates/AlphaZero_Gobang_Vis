import numpy as np
import pygame
import sys
import traceback
import copy
from pygame.locals import *

#pathname = '/Users/yuyongsheng/Downloads/'

pygame.init()
pygame.mixer.init()

#颜色
background=(201,202,187)
checkerboard=(80,80,80)
button=(52,53,44)



#音乐
# play_chess_sound = pygame.mixer.Sound("music/play_chess.wav")
# play_chess_sound.set_volume(0.2)
# button_sound = pygame.mixer.Sound("music/button.wav")
# button_sound.set_volume(0.2)
# victor_sound = pygame.mixer.Sound("music/victory.wav")
# victor_sound.set_volume(0.2)

#绘制棋盘
def Draw_a_chessboard(screen):  
    #填充背景色
    screen.fill(background)
    #Background=pygame.image.load("background.jpg").convert_alpha()
    #screen.blit(Background,(0,0))
    #画棋盘
    for i in range(21):
        pygame.draw.line(screen, checkerboard, (40*i+3, 3), (40*i+3, 803)) 
        pygame.draw.line(screen, checkerboard, (3, 40*i+3), (803, 40*i+3))
    #画边线
    pygame.draw.line(screen, checkerboard, (3, 3), (803, 3),5)   
    pygame.draw.line(screen, checkerboard, (3, 3), (3, 803),5)   
    pygame.draw.line(screen, checkerboard, (803, 3), (803, 803),5)   
    pygame.draw.line(screen, checkerboard, (3, 803), (803, 803),5) 
    
    #画定位点
    pygame.draw.circle(screen, checkerboard, (163, 163), 6) 
    pygame.draw.circle(screen, checkerboard, (163, 643), 6) 
    pygame.draw.circle(screen, checkerboard, (643, 163), 6) 
    pygame.draw.circle(screen, checkerboard, (643, 643), 6) 
    pygame.draw.circle(screen, checkerboard, (403, 403), 6) 
    
    #画‘悔棋’‘重新开始’跟‘退出’按钮
    pygame.draw.rect(screen,button,[900,350,120,100],5)
    pygame.draw.rect(screen,button,[900,500,200,100],5)
    pygame.draw.rect(screen,button,[900,650,200,100],5)
    s_font = pygame.font.Font(pathname+'STFANGSO.ttf', 40)
    text1=s_font.render("悔棋",True,button)
    text2=s_font.render("重新开始",True,button)
    text3=s_font.render("退出游戏",True,button)
    screen.blit(text1,(920,370))
    screen.blit(text2,(920,520))
    screen.blit(text3,(920,670))

#绘制棋子（横坐标，纵坐标，屏幕，棋子颜色（1代表黑，2代表白））
def Draw_a_chessman(x,y,screen,color):    
    if color==1:        
        Black_chess=pygame.image.load(pathname+"Black_chess.png").convert_alpha()
        screen.blit(Black_chess,(40*x+3-15,40*y+3-15))
    if color==2:
        White_chess=pygame.image.load(pathname+"White_chess.png").convert_alpha()
        screen.blit(White_chess,(40*x+3-15,40*y+3-15))

#绘制带有棋子的棋盘
def Draw_a_chessboard_with_chessman(map,screen):  
    screen.fill(background)
    Draw_a_chessboard(screen)
    for i in range(24):
        for j in range(24):
            Draw_a_chessman(i+1,j+1,screen,map[i][j])



#定义存储棋盘的列表,
#列表为24列24行是因为判断是否胜利函数里的索引会超出19
#列表大一点不会对游戏有什么影响
map=[]
for i in range(24):
    map.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#清零map列表
def clear():
    global map
    for i in range(24):
        for j in range(24):
            map[i][j]=0

#判断是否胜利
def win(i, j):
    k = map[i][j]
    p=[]
    for a in range(20):
        p.append(0)
    for i3 in range(i-4,i+5):
        for j3 in range(j-4,j+5):
            if (map[i3][j3] == k and i3 - i == j3 - j and i3 <= i and j3 <= j):
                p[0]+=1
            if (map[i3][j3] == k and j3 == j and i3 <= i and j3 <= j):
                p[1]+=1
            if (map[i3][j3] == k and i3 == i and i3 <= i and j3 <= j):
                p[2]+=1
            if (map[i3][j3] == k and i3 - i == j3 - j and i3 >= i and j3 >= j):
                p[3]+=1
            if (map[i3][j3] == k and j3 == j and i3 >= i and j3 >= j):
                p[4]+=1
            if (map[i3][j3] == k and i3 == i and i3 >= i and j3 >= j):
                p[5]+=1
            if (map[i3][j3] == k and i - i3 == j3 - j and i3 <= i and j3 >= j):
                p[6]+=1
            if (map[i3][j3] == k and i3 - i == j - j3 and i3 >= i and j3 <= j):
                p[7]+=1
            if (map[i3][j3] == k and j - j3 == i - i3 and i3 <= i + 1  and  i3 >= i - 3  and  j3 <= j + 1  and  j3 >= j - 3):
                p[8]+=1
            if (map[i3][j3] == k and j == j3 and i3 <= i + 1  and  i3 >= i - 3  and  j3 <= j + 1  and  j3 >= j - 3):
                p[9]+=1
            if (map[i3][j3] == k and i == i3 and i3 <= i + 1  and  i3 >= i - 3  and  j3 <= j + 1  and  j3 >= j - 3):
                p[10]+=1
            if (map[i3][j3] == k and j - j3 == i - i3 and i3 >= i - 1  and  i3 <= i + 3  and  j3 >= j - 1  and  j3 <= j + 3):
                p[11]+=1
            if (map[i3][j3] == k and j == j3 and i3 >= i - 1  and  i3 <= i + 3  and  j3 >= j - 1  and  j3 <= j + 3):
                p[12]+=1
            if (map[i3][j3] == k and i == i3 and i3 >= i - 1  and  i3 <= i + 3  and  j3 >= j - 1  and  j3 <= j + 3):
                p[13]+=1
            if (map[i3][j3] == k and i - i3 == j3 - j and i3 <= i + 1  and  i3 >= i - 3  and  j3 >= j - 1  and  j3 <= j + 3):
                p[14]+=1
            if (map[i3][j3] == k and i3 - i == j - j3 and i3 >= i - 1  and  i3 <= i + 3  and  j3 <= j + 1  and  j3 >= j - 3):
                p[15]+=1
            if (map[i3][j3] == k and j - j3 == i - i3 and i3 <= i + 2  and  i3 >= i - 2  and  j3 <= j + 2  and  j3 >= j - 2):
                p[16]+=1
            if (map[i3][j3] == k and j == j3 and i3 <= i + 2  and  i3 >= i - 2  and  j3 <= j + 2  and  j3 >= j - 2):
                p[17]+=1
            if (map[i3][j3] == k and i == i3 and i3 <= i + 2  and  i3 >= i - 2  and  j3 <= j + 2  and  j3 >= j - 2):
                p[18]+=1
            if (map[i3][j3] == k and i - i3 == j3 - j and i3 <= i + 2  and  i3 >= i - 2  and  j3 <= j + 2  and  j3 >= j - 2):
                p[19]+=1
    for b in range(20):
        if p[b]==5:
            return True
    return False

#绘制提示器（类容，屏幕，字大小）
def text(s,screen,x):
    #先把上一次的类容用一个矩形覆盖
    pygame.draw.rect(screen,background,[850,100,1200,100])
    #定义字体跟大小
    s_font=pygame.font.Font(pathname+'STFANGSO.ttf',x)
    #定义类容，是否抗锯齿，颜色
    s_text=s_font.render(s,True,button)
    #将字放在窗口指定位置
    screen.blit(s_text,(880,100))
    pygame.display.flip()

#用于控制顺序
t=True

#用于结束游戏后阻止落子
running=True

#主函数
def main():
    #将 t，map，running设置为可改的
    global t,map,running,maps,r,h
    #将map置零
    clear()
    #定义储存所有棋盘状态的列表（用于悔棋）
    map2=copy.deepcopy(map)
    maps=[map2]

    
    #定义窗口
    screen = pygame.display.set_mode([1200,806])
    
    #定义窗口名字
    pygame.display.set_caption("五子棋")
    
    #在窗口画出棋盘，提示器以及按钮
    Draw_a_chessboard(screen)
    pygame.display.flip()
    clock=pygame.time.Clock()
    while True:
        #只有running为真才能落子，主要用于游戏结束后防止再次落子
        if running:
            if t:
                color=1
                text('黑棋落子',screen,54)
            else:
                color=2
                text('白棋落子',screen,54)
        
        for event in pygame.event.get():
            #点击x则关闭窗口
            if event.type ==pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            #点击窗口里面类容则完成相应指令
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    x,y=event.pos[0],event.pos[1]
                    for i in range(19):
                        for j in range(19):
                            #点击棋盘相应位置
                            if i*40+3+20<x<i*40+3+60 and j*40+3+20<y<j*40+3+60 and not map[i][j] and running:
                                #在棋盘相应位置落相应颜色棋子
                                Draw_a_chessman(i+1,j+1,screen,color)
                                #播放音效
                                #play_chess_sound.play(0)
                                #在map里面记录落子位置
                                map[i][j]=color

                                #将map存入maps
                                map3=copy.deepcopy(map)
                                maps.append(map3)

                                #判断落子后是否有五子一线
                                if win(i,j):
                                    if t:
                                        text('黑棋胜利，请重新游戏',screen,30)
                                    else:
                                        text('白棋胜利，请重新游戏',screen,30)
                                    #播放音效
                                    #victor_sound.play(0)
                                    #阻止再往棋盘落子
                                    running=False
                                pygame.display.flip()
                                t=not t
                    #如果点击‘重新开始’
                    if 900<x<1100 and 500<y<600:
                        #取消阻止
                        running=True
                        #播放音效
                        #button_sound.play(0)
                        #重新开始
                        main()
                    
                    #点击‘退出游戏’，退出游戏
                    elif 900<x<1100 and 650<y<750:
                        #播放音效
                        #button_sound.play(0)
                        pygame.quit()
                        sys.exit()
 
                    #点击‘悔棋’
                    elif 900<x<1020 and 350<y<450 and len(maps)!=1:
                        #播放音效
                        #button_sound.play(0)
                        #删除maps里最后一个元素
                        del maps[len(maps)-1] 
                        #再将最后一个元素copy给map
                        map=copy.deepcopy(maps[len(maps)-1])
                        #切换顺序
                        t=not t
                        #将map显示出来
                        Draw_a_chessboard_with_chessman(map,screen)
                        #悔棋完成，阻止再次悔棋
                        x,y=0,0
        clock.tick(60)
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
        input()

