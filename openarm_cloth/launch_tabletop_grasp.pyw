"""Local scene selector for the colour-independent geometry-grasp demo."""
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk,messagebox

ROOT=Path(__file__).resolve().parent


def build_launch_command(executable,report,*,fixture=None):
    """Build scene-setup arguments; automatic mode sends no shape selection."""
    executable=Path(executable)
    if executable.name.lower()=='pythonw.exe': executable=executable.with_name('python.exe')
    command=[str(executable),'-u',str(ROOT/'demo_tabletop_objects.py'),'--report',str(report)]
    if fixture is None:
        return command+['--random-fixture']
    shape,colour,size,position=fixture
    return command+['--shape',shape,'--color',colour,'--size',*map(str,size),
                    '--fixture-pose',*map(str,position)]


def main():
    window=tk.Tk();window.title('物体自动视觉识别与抓取');window.resizable(False,False)
    form=ttk.Frame(window,padding=18);form.grid()
    custom=tk.BooleanVar(value=False)
    shapes={'方块':('cube',[57.6,57.6,57.6]),'长方体':('box',[70,40,55]),'圆柱':('cylinder',[45,45,75])}
    colours={'蓝色':'blue','绿色':'green','黑色':'black','白色':'white','红色':'red','黄色':'yellow','棕色':'brown'}
    shape=tk.StringVar(value='圆柱');colour=tk.StringVar(value='蓝色')
    dimensions=[tk.StringVar(value=str(v)) for v in shapes['圆柱'][1]]
    pose=[tk.StringVar(value=v) for v in ('0.43','-0.10','15')]
    ttk.Label(form,text='自动识别：无需先选形状、颜色或尺寸。\n相机测量方块 / 长方体 / 圆柱，再进行 IK 规划。').grid(row=0,column=0,columnspan=3,pady=(0,8))
    fixture_form=ttk.Frame(form);fixture_form.grid(row=2,column=0,columnspan=3,sticky='ew')
    def toggle_fixture():
        if custom.get(): fixture_form.grid()
        else: fixture_form.grid_remove()
    ttk.Checkbutton(form,text='自定义仿真测试物体（仅布置场景，不给识别器答案）',
                    variable=custom,command=toggle_fixture).grid(row=1,column=0,columnspan=3,pady=(0,8))
    ttk.Label(fixture_form,text='形状').grid(row=1,column=0,sticky='w')
    shape_box=ttk.Combobox(fixture_form,textvariable=shape,values=list(shapes),state='readonly',width=16)
    shape_box.grid(row=1,column=1,columnspan=2,sticky='ew')
    def preset(event=None):
        for variable,value in zip(dimensions,shapes[shape.get()][1]): variable.set(str(value))
    shape_box.bind('<<ComboboxSelected>>',preset)
    ttk.Label(fixture_form,text='颜色').grid(row=2,column=0,sticky='w')
    ttk.Combobox(fixture_form,textvariable=colour,values=list(colours),state='readonly',width=16).grid(row=2,column=1,columnspan=2,sticky='ew')
    for i,label in enumerate(('X尺寸 / mm','Y尺寸 / mm','高度 / mm')):
        ttk.Label(fixture_form,text=label).grid(row=3+i,column=0,sticky='w')
        ttk.Entry(fixture_form,textvariable=dimensions[i],width=18).grid(row=3+i,column=1,columnspan=2,sticky='ew')
    for i,label in enumerate(('初始 X / m','初始 Y / m','旋转 / 度')):
        ttk.Label(fixture_form,text=label).grid(row=6+i,column=0,sticky='w')
        ttk.Entry(fixture_form,textvariable=pose[i],width=18).grid(row=6+i,column=1,columnspan=2,sticky='ew')
    toggle_fixture()
    status=tk.StringVar(value='默认随机布置一个受支持的测试物体，识别器不接收其类型。\n胸部粗定位 → 腕部精定位 → IK → 物理夹取。')
    ttk.Label(form,textvariable=status,wraplength=440).grid(row=4,column=0,columnspan=3,pady=12)
    running={}
    def launch():
        try:
            if custom.get():
                size=[float(v.get())/1000 for v in dimensions];position=[float(v.get()) for v in pose]
                if not all(math.isfinite(v) for v in size+position): raise ValueError('参数必须是有限数字')
                if min(size)<=0: raise ValueError('尺寸必须大于零')
                if shape.get()=='圆柱' and abs(size[0]-size[1])>1e-8: raise ValueError('圆柱的 X、Y 直径应相等')
        except ValueError as error: messagebox.showerror('参数无效',str(error));return
        folder=ROOT/'outputs'/'generic_objects'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
        report=folder/'report.json';log=(folder/'run.log').open('w',encoding='utf-8')
        fixture=(shapes[shape.get()][0],colours[colour.get()],size,position) if custom.get() else None
        command=build_launch_command(sys.executable,report,fixture=fixture)
        try: process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception as error: log.close();messagebox.showerror('启动失败',str(error));return
        running.update(process=process,log=log,report=report)
        start.configure(state='disabled');status.set('青色挡边、黄色边线内摆放；摆放阶段限制完整外形不越界。松开鼠标再按空格，开始后不锁位置。R 恢复，Q 退出。')
        window.after(500,poll)
    def poll():
        if running['process'].poll() is None: window.after(500,poll);return
        running['log'].close();report=running['report'];start.configure(state='normal')
        if report.exists():
            result=json.loads(report.read_text(encoding='utf-8'))
            status.set('本次结果：'+result.get('verdict','UNKNOWN')+'。'+result.get('error','')+' 报告：'+str(report))
        else: status.set('运行未完成，请检查日志：'+str(report.with_name('run.log')))
    start=ttk.Button(form,text='打开场景，由相机自动识别',command=launch)
    start.grid(row=3,column=0,columnspan=3,pady=(14,0))
    window.mainloop()


if __name__=='__main__': main()
