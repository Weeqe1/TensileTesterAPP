import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import savgol_filter
import numpy as np
import os

# 设置外观
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TensileTesterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("拉力计数据处理器 - TensileTesterAPP v2.1")
        self.geometry("1100x700")

        # 数据变量
        self.raw_df = None
        self.processed_df = None
        self.file_path = ""
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
        plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

        self.setup_ui()

    def setup_ui(self):
        # 配置网格布局
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 左侧控制面板
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="数据处理中心", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=(20, 20))

        # 文件选择
        self.btn_select_file = ctk.CTkButton(self.sidebar, text="选择数据文件 (CSV)", command=self.select_file)
        self.btn_select_file.pack(pady=10, padx=20)
        
        self.lbl_file_status = ctk.CTkLabel(self.sidebar, text="未选择文件", font=ctk.CTkFont(size=12))
        self.lbl_file_status.pack(pady=(0, 10))

        # 参数设置
        self.param_frame = ctk.CTkFrame(self.sidebar)
        self.param_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(self.param_frame, text="数值过滤范围 (N)").pack(pady=5)
        
        self.range_inner_frame = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.range_inner_frame.pack()
        
        self.entry_min = ctk.CTkEntry(self.range_inner_frame, placeholder_text="下限", width=80)
        self.entry_min.pack(side="left", padx=5, pady=5)
        self.entry_min.insert(0, "0")

        self.entry_max = ctk.CTkEntry(self.range_inner_frame, placeholder_text="上限", width=80)
        self.entry_max.pack(side="left", padx=5, pady=5)
        self.entry_max.insert(0, "100")

        ctk.CTkLabel(self.param_frame, text="时间过滤范围 (S)").pack(pady=5)
        
        self.time_range_frame = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.time_range_frame.pack()
        
        self.entry_time_min = ctk.CTkEntry(self.time_range_frame, placeholder_text="下限", width=80)
        self.entry_time_min.pack(side="left", padx=5, pady=5)
        self.entry_time_min.insert(0, "all")

        self.entry_time_max = ctk.CTkEntry(self.time_range_frame, placeholder_text="上限", width=80)
        self.entry_time_max.pack(side="left", padx=5, pady=5)
        self.entry_time_max.insert(0, "all")

        ctk.CTkLabel(self.param_frame, text="摩擦系数").pack(pady=5)
        self.entry_friction = ctk.CTkEntry(self.param_frame, placeholder_text="摩擦系数 (默认1.0)")
        self.entry_friction.pack(pady=5, padx=20)
        self.entry_friction.insert(0, "1.0")

        # 平滑设置区域（带帮助图标）
        self.smooth_label_frame = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.smooth_label_frame.pack(pady=(5, 0))
        
        ctk.CTkLabel(self.smooth_label_frame, text="平滑处理窗口大小").pack(side="left")
        self.btn_help = ctk.CTkButton(self.smooth_label_frame, text="?", width=20, height=20, 
                                     corner_radius=10, fg_color="gray", hover_color="#555555",
                                     command=self.show_smooth_help)
        self.btn_help.pack(side="left", padx=5)

        self.entry_smooth = ctk.CTkEntry(self.param_frame, placeholder_text="窗口大小 (奇数)")
        self.entry_smooth.pack(pady=5, padx=20)
        self.entry_smooth.insert(0, "51")

        # 功能按钮
        self.btn_process = ctk.CTkButton(self.sidebar, text="执行计算与可视化", fg_color="#1f6aa5", hover_color="#144870", command=self.process_data)
        self.btn_process.pack(pady=20, padx=20)

        self.btn_save_csv = ctk.CTkButton(self.sidebar, text="保存数据表 (.csv)", state="disabled", command=self.save_csv)
        self.btn_save_csv.pack(pady=10, padx=20)

        self.btn_save_plot = ctk.CTkButton(self.sidebar, text="保存可视化图表 (.png)", state="disabled", command=self.save_plot)
        self.btn_save_plot.pack(pady=10, padx=20)

        # 数据统计显示
        self.stats_frame = ctk.CTkFrame(self.sidebar, fg_color="#333333")
        self.stats_frame.pack(fill="x", padx=15, pady=20)
        
        ctk.CTkLabel(self.stats_frame, text="数据统计 (加工后)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.stats_inner = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        self.stats_inner.pack(pady=5)
        
        ctk.CTkLabel(self.stats_inner, text="平均拉力 (Force):").pack(side="left", padx=5)
        self.lbl_avg_force = ctk.CTkLabel(self.stats_inner, text="-- N", text_color="#ffcc00", font=ctk.CTkFont(weight="bold"))
        self.lbl_avg_force.pack(side="left", padx=5)

        # 右侧显示区域
        self.main_content = ctk.CTkFrame(self)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.canvas_frame = ctk.CTkFrame(self.main_content)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.lbl_file_status.configure(text=os.path.basename(file_path))
            try:
                # 寻找真实的表头行，跳过前导无用信息
                skip = 0
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            line_lower = line.lower()
                            if 'time' in line_lower or 'load' in line_lower or 'force' in line_lower or 'reading' in line_lower or '时间' in line_lower or '力' in line_lower:
                                skip = i
                                break
                except Exception:
                    pass
                
                # 加载对应数据
                try:
                    self.raw_df = pd.read_csv(file_path, skiprows=skip)
                except UnicodeDecodeError:
                    self.raw_df = pd.read_csv(file_path, skiprows=skip, encoding='gbk')
                    
                messagebox.showinfo("成功", f"文件已加载，共 {len(self.raw_df)} 行数据")
            except Exception as e:
                messagebox.showerror("错误", f"读取文件失败: {e}")

    def process_data(self):
        if self.raw_df is None:
            messagebox.showwarning("警告", "请先选择数据文件")
            return

        try:
            df = self.raw_df.copy()
            
            # 智能匹配时间与力值列
            cols = df.columns.tolist()
            time_col = None
            force_col = None
            
            for col in cols:
                col_lower = str(col).lower()
                if ('time' in col_lower or '时间' in col_lower or 'sec' in col_lower) and time_col is None:
                    time_col = col
                elif ('load' in col_lower or 'force' in col_lower or '力' in col_lower) and force_col is None:
                    force_col = col
                    
            if time_col is None:
                time_col = cols[0] 
            if force_col is None:
                force_col = cols[1] if len(cols) > 1 else cols[0]

            # 清洗字符串干扰数据（转为数值型，无法转换的剔除）
            df[force_col] = pd.to_numeric(df[force_col], errors='coerce')
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.dropna(subset=[force_col, time_col])

            # 1. 负值统一转正值
            df[force_col] = df[force_col].abs()

            # 2. 只有关注的数值范围 (力 + 时间)
            try:
                min_v = float(self.entry_min.get())
                max_v = float(self.entry_max.get())
                
                # 时间范围过滤逻辑
                t_min_str = self.entry_time_min.get().strip().lower()
                t_max_str = self.entry_time_max.get().strip().lower()
                
                t_min = -float('inf') if t_min_str == "all" or t_min_str == "" else float(t_min_str)
                t_max = float('inf') if t_max_str == "all" or t_max_str == "" else float(t_max_str)
                
                # 摩擦系数
                friction_str = self.entry_friction.get().strip()
                friction_coeff = float(friction_str) if friction_str else 1.0
                if friction_coeff == 0:
                    messagebox.showerror("错误", "摩擦系数不能为 0")
                    return
                
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")
                return

            df = df[(df[force_col] >= min_v) & (df[force_col] <= max_v)]
            df = df[(df[time_col] >= t_min) & (df[time_col] <= t_max)]

            if df.empty:
                messagebox.showwarning("提示", "当前范围内无数据，请调整范围")
                return

            # 4. 摩擦系数处理
            df['Friction coefficient processing'] = df[force_col] / friction_coeff
            target_col = 'Friction coefficient processing'

            # 5. 平滑处理
            try:
                window_size = int(self.entry_smooth.get())
                if window_size % 2 == 0:
                    window_size += 1 # Savitzky-Golay 需要奇数
                
                if window_size > len(df):
                    window_size = len(df) if len(df) % 2 != 0 else len(df) - 1
                
                if window_size < 3:
                    df['Smooth_Data'] = df[target_col]
                else:
                    df['Smooth_Data'] = savgol_filter(df[target_col], window_size, 3)
            except Exception as e:
                messagebox.showerror("平滑处理错误", f"平滑处理失败: {e}")
                df['Smooth_Data'] = df[target_col]

            # 6. 计算统计值 (处理后的 Force 平均值)
            avg_force = df[force_col].mean()
            self.lbl_avg_force.configure(text=f"{avg_force:.4f} N")

            self.processed_df = df
            self.update_plot(time_col, target_col)
            
            self.btn_save_csv.configure(state="normal")
            self.btn_save_plot.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror("计算错误", f"处理过程中发生错误: {e}")

    def update_plot(self, time_col, data_col):
        self.ax.clear()
        
        # 绘制原始点（淡化）
        self.ax.scatter(self.processed_df[time_col], self.processed_df[data_col], color='lightgray', s=1, label='处理后的原始数据')
        
        # 绘制平滑曲线
        self.ax.plot(self.processed_df[time_col], self.processed_df['Smooth_Data'], color='red', linewidth=2, label='平滑后的数据')
        
        self.ax.set_title("摩擦系数处理后随时间变化图")
        self.ax.set_xlabel(f"时间 ({time_col})")
        self.ax.set_ylabel("加工后的力值 (Processed)")
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def show_smooth_help(self):
        help_text = (
            "【平滑处理窗口大小说明】\n\n"
            "1. 含义：决定了平滑时参考的邻近数据点数量。\n\n"
            "2. 数值大小的影响：\n"
            "   - 【较小值 (如 5-21)】：保留更多细节和突变建议，但去噪能力弱，曲线可能仍有毛刺。\n"
            "   - 【较大值 (如 51-101)】：曲线更平滑，能反映整体趋势，但可能导致峰值被压扁，丢失特征。\n\n"
            "3. 如何设置：\n"
            "   - 对于采样率高的拉力测试，通常在 31-101 效果最佳。\n"
            "   - 必须是正奇数（程序会自动将偶数转为奇数）。\n"
            "   - 建议先从 51 开始尝试。"
        )
        messagebox.showinfo("参数设置帮助与建议", help_text)

    def save_csv(self):
        if self.processed_df is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if save_path:
                self.processed_df.to_csv(save_path, index=False)
                messagebox.showinfo("成功", "数据表已成功保存")

    def save_plot(self):
        if self.fig is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg")])
            if save_path:
                self.fig.savefig(save_path)
                messagebox.showinfo("成功", "可视化图表已成功保存")

if __name__ == "__main__":
    app = TensileTesterApp()
    app.mainloop()
