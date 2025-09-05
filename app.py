import gradio as gr
import os
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import uuid
import itertools

class EloRating:
    def __init__(self, k=32):
        self.k = k
        self.ratings_file = 'ratings.json'
        self.history_file = 'history.json'
        self.config_file = 'config.json'
        self.sessions_file = 'sessions.json'
        self.ratings = self.load_ratings()
        self.history = self.load_history()
        self.config = self.load_config()
        self.sessions = self.load_sessions()

    def load_ratings(self):
        if os.path.exists(self.ratings_file):
            try:
                with open(self.ratings_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"警告：无法加载评分文件 {self.ratings_file}: {e}")
                print("将使用空的评分数据")
                return {}
        return {}

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        return []
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"警告：无法加载历史文件 {self.history_file}: {e}")
                print("将使用空的历史数据")
                return []
        return []

    def load_config(self):
        default_config = {
            'a_dir': 'a',
            'b_dir': 'b', 
            'caption_dir': 'caption',
            'k_factor': 32,
            'initial_rating': 1500
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        return default_config
                    config = json.loads(content)
                    return {**default_config, **config}
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"警告：无法加载配置文件 {self.config_file}: {e}")
                print("将使用默认配置")
                return default_config
        return default_config

    def load_sessions(self):
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"警告：无法加载会话文件 {self.sessions_file}: {e}")
                print("将使用空的会话数据")
                return {}
        return {}

    def save_ratings(self):
        with open(self.ratings_file, 'w', encoding='utf-8') as f:
            json.dump(self.ratings, f, indent=2, ensure_ascii=False)

    def save_history(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def save_config(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def save_sessions(self):
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(self.sessions, f, indent=2, ensure_ascii=False)

    def create_session(self):
        """创建新的测试会话"""
        session_id = str(uuid.uuid4())
        
        # 加载所有图片
        config = self.config
        a_images, b_images, captions = load_image_pairs(
            config['a_dir'], config['b_dir'], config['caption_dir']
        )
        
        if not a_images or not b_images:
            return None, "请确保a和b文件夹中都有图片"
        
        # 按文件名（不含扩展名）匹配图片对
        matched_pairs = []
        
        # 创建文件名到完整路径的映射
        a_name_to_path = {Path(img).stem: img for img in a_images}
        b_name_to_path = {Path(img).stem: img for img in b_images}
        
        # 找到共同的文件名
        common_names = set(a_name_to_path.keys()) & set(b_name_to_path.keys())
        
        if not common_names:
            return None, "a和b文件夹中没有找到匹配的文件名（不含扩展名）"
        
        # 为每个匹配的文件名创建图片对
        for name in common_names:
            img_a = a_name_to_path[name]
            img_b = b_name_to_path[name]
            
            # 获取对应的caption
            caption = captions.get(name, f"请选择更好的图片 ({name})")
            
            matched_pairs.append({
                'a': img_a,
                'b': img_b,
                'caption': caption,
                'name': name  # 添加文件名用于调试
            })
        
        # 随机打乱顺序
        random.shuffle(matched_pairs)
        
        # 保存会话信息
        self.sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'pairs': matched_pairs,
            'current_index': 0,
            'completed': False,
            'total_pairs': len(matched_pairs)
        }
        
        self.save_sessions()
        return session_id, f"已创建新会话，找到{len(matched_pairs)}个匹配的图片对待测试"

    def get_next_pair(self, session_id):
        """获取下一个图片对"""
        if session_id not in self.sessions:
            return None, None, "会话不存在", "会话已过期，请刷新页面"
        
        session = self.sessions[session_id]
        
        if session['completed'] or session['current_index'] >= len(session['pairs']):
            return None, None, "测试完成！", f"🎉 恭喜！您已完成所有{session['total_pairs']}个图片对的测试！"
        
        current_pair = session['pairs'][session['current_index']]
        progress = f"进度: {session['current_index'] + 1}/{session['total_pairs']} ({(session['current_index'] + 1)/session['total_pairs']*100:.1f}%)"
        
        return current_pair['a'], current_pair['b'], current_pair['caption'], progress

    def vote_and_next(self, session_id, choice):
        """投票并获取下一个图片对"""
        if session_id not in self.sessions:
            return None, None, "", "会话不存在"
        
        session = self.sessions[session_id]
        
        if session['completed'] or session['current_index'] >= len(session['pairs']):
            return None, None, "", "测试已完成"
        
        # 获取当前图片对
        current_pair = session['pairs'][session['current_index']]
        
        # 记录投票
        if choice == "A":
            winner, loser = current_pair['a'], current_pair['b']
        else:
            winner, loser = current_pair['b'], current_pair['a']
        
        r1, r2 = self.update_rating(winner, loser)
        
        # 移动到下一个图片对
        session['current_index'] += 1
        
        # 检查是否完成
        if session['current_index'] >= len(session['pairs']):
            session['completed'] = True
            self.save_sessions()
            return None, None, "🎉 测试完成！", f"已完成所有{session['total_pairs']}个图片对的测试！获胜图片评分: {r1:.1f}, 失败图片评分: {r2:.1f}"
        
        self.save_sessions()
        
        # 获取下一个图片对
        next_pair = session['pairs'][session['current_index']]
        progress = f"进度: {session['current_index'] + 1}/{session['total_pairs']} ({(session['current_index'] + 1)/session['total_pairs']*100:.1f}%)"
        result_msg = f"已记录选择！获胜图片评分: {r1:.1f}, 失败图片评分: {r2:.1f}. {progress}"
        
        return next_pair['a'], next_pair['b'], next_pair['caption'], result_msg

    def get_rating(self, image_path):
        return self.ratings.get(image_path, self.config['initial_rating'])

    def update_rating(self, winner, loser):
        r1 = self.get_rating(winner)
        r2 = self.get_rating(loser)

        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

        r1_new = r1 + self.config['k_factor'] * (1 - e1)
        r2_new = r2 + self.config['k_factor'] * (0 - e2)

        self.ratings[winner] = r1_new
        self.ratings[loser] = r2_new

        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "winner": winner,
            "loser": loser,
            "winner_rating_before": r1,
            "loser_rating_before": r2,
            "winner_rating_after": r1_new,
            "loser_rating_after": r2_new,
            "rating_change": r1_new - r1
        })

        self.save_ratings()
        self.save_history()

        return r1_new, r2_new

    def get_statistics(self):
        if not self.ratings:
            return {}
        
        # 按模型分组统计
        model_stats = {}
        for img_path, rating in self.ratings.items():
            # 假设文件名包含模型信息，或者根据文件夹判断
            if '/a/' in img_path or '\\a\\' in img_path:
                model = 'Model A'
            elif '/b/' in img_path or '\\b\\' in img_path:
                model = 'Model B'
            else:
                model = 'Unknown'
            
            if model not in model_stats:
                model_stats[model] = {'ratings': [], 'wins': 0, 'losses': 0}
            
            model_stats[model]['ratings'].append(rating)
        
        # 统计胜负
        for record in self.history:
            winner_model = 'Model A' if ('/a/' in record['winner'] or '\\a\\' in record['winner']) else 'Model B'
            loser_model = 'Model A' if ('/a/' in record['loser'] or '\\a\\' in record['loser']) else 'Model B'
            
            if winner_model in model_stats:
                model_stats[winner_model]['wins'] += 1
            if loser_model in model_stats:
                model_stats[loser_model]['losses'] += 1
        
        # 计算平均评分和胜率
        for model in model_stats:
            ratings = model_stats[model]['ratings']
            model_stats[model]['avg_rating'] = np.mean(ratings) if ratings else 0
            model_stats[model]['std_rating'] = np.std(ratings) if ratings else 0
            
            total_games = model_stats[model]['wins'] + model_stats[model]['losses']
            model_stats[model]['win_rate'] = model_stats[model]['wins'] / total_games if total_games > 0 else 0
        
        return model_stats

    def export_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出评分数据
        ratings_df = pd.DataFrame([
            {'image_path': path, 'rating': rating}
            for path, rating in self.ratings.items()
        ])
        ratings_file = f'ratings_export_{timestamp}.csv'
        ratings_df.to_csv(ratings_file, index=False, encoding='utf-8-sig')
        
        # 导出历史数据
        if self.history:
            history_df = pd.DataFrame(self.history)
            history_file = f'history_export_{timestamp}.csv'
            history_df.to_csv(history_file, index=False, encoding='utf-8-sig')
        
        return f"数据已导出: {ratings_file}" + (f", {history_file}" if self.history else "")

    def reset_data(self):
        self.ratings = {}
        self.history = []
        self.sessions = {}
        self.save_ratings()
        self.save_history()
        self.save_sessions()

def load_image_pairs(a_dir, b_dir, caption_dir):
    a_images = [str(p) for p in Path(a_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]]
    b_images = [str(p) for p in Path(b_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]]
    
    # 获取对应的标题
    captions = {}
    if os.path.exists(caption_dir):
        for p in Path(caption_dir).glob("*.txt"):
            base_name = p.stem
            with open(p, 'r', encoding='utf-8') as f:
                captions[base_name] = f.read().strip()
    
    return a_images, b_images, captions

def create_rating_plot(elo_system):
    if not elo_system.history:
        return None
    
    # 创建评分变化图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 评分历史趋势
    timestamps = [datetime.fromisoformat(h['timestamp']) for h in elo_system.history]
    winner_ratings = [h['winner_rating_after'] for h in elo_system.history]
    loser_ratings = [h['loser_rating_after'] for h in elo_system.history]
    
    ax1.plot(timestamps, winner_ratings, 'g-', label='获胜者评分', alpha=0.7)
    ax1.plot(timestamps, loser_ratings, 'r-', label='失败者评分', alpha=0.7)
    ax1.set_title('评分变化趋势')
    ax1.set_ylabel('ELO评分')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 评分分布
    all_ratings = list(elo_system.ratings.values())
    if all_ratings:
        ax2.hist(all_ratings, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('当前评分分布')
        ax2.set_xlabel('ELO评分')
        ax2.set_ylabel('图片数量')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 初始化ELO系统
elo_system = EloRating()

# 全局变量
current_session_id = None

def start_new_session():
    """开始新的测试会话"""
    global current_session_id
    session_id, message = elo_system.create_session()
    if session_id:
        current_session_id = session_id
        img_a, img_b, caption, progress = elo_system.get_next_pair(session_id)
        return img_a, img_b, caption, f"✅ {message}\n{progress}"
    else:
        return None, None, "", f"❌ {message}"

def vote(choice):
    """投票函数"""
    global current_session_id
    if not current_session_id:
        return None, None, "", "请先开始新的测试会话"
    
    return elo_system.vote_and_next(current_session_id, choice)

def get_current_pair():
    """获取当前图片对"""
    global current_session_id
    if not current_session_id:
        return None, None, "", "请先开始新的测试会话"
    
    return elo_system.get_next_pair(current_session_id)

def get_statistics_display():
    stats = elo_system.get_statistics()
    if not stats:
        return "暂无统计数据"
    
    display_text = "## 📊 模型统计\n\n"
    
    for model, data in stats.items():
        display_text += f"### {model}\n"
        display_text += f"- **平均评分**: {data['avg_rating']:.1f} (±{data['std_rating']:.1f})\n"
        display_text += f"- **胜率**: {data['win_rate']:.1%}\n"
        display_text += f"- **胜场**: {data['wins']} | **败场**: {data['losses']}\n"
        display_text += f"- **图片数量**: {len(data['ratings'])}\n\n"
    
    return display_text

def export_data():
    return elo_system.export_data()

def reset_all_data():
    global current_session_id
    elo_system.reset_data()
    current_session_id = None
    return "✅ 所有数据已重置"

def update_config(a_dir, b_dir, caption_dir, k_factor, initial_rating):
    elo_system.config.update({
        'a_dir': a_dir,
        'b_dir': b_dir,
        'caption_dir': caption_dir,
        'k_factor': k_factor,
        'initial_rating': initial_rating
    })
    elo_system.save_config()
    return "✅ 配置已保存"

# 启动界面
with gr.Blocks(title="ELO图片评分系统 - 完整测试版", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 ELO图片质量评估系统 - 完整测试版")
    gr.Markdown("通过双盲对比评估不同模型生成的图片质量，确保每次会话完整测试所有图片对")
    
    with gr.Tabs():
        # 图片对比标签页
        with gr.Tab("🖼️ 图片对比"):
            with gr.Row():
                with gr.Column():
                    img_a = gr.Image(label="图片 A", height=400)
                    btn_a = gr.Button("选择 A", variant="primary", size="lg")
                
                with gr.Column():
                    img_b = gr.Image(label="图片 B", height=400)
                    btn_b = gr.Button("选择 B", variant="primary", size="lg")
            
            caption_text = gr.Textbox(label="提示文本", interactive=False)
            result = gr.Textbox(label="评分结果和进度", interactive=False)
            
            with gr.Row():
                start_session_btn = gr.Button("🚀 开始新的完整测试", variant="primary")
                skip_btn = gr.Button("⏭️ 跳过这对", variant="secondary")
        
        # 统计分析标签页
        with gr.Tab("📊 统计分析"):
            with gr.Row():
                with gr.Column():
                    stats_display = gr.Markdown(get_statistics_display())
                    refresh_stats_btn = gr.Button("🔄 刷新统计", variant="secondary")
                
                with gr.Column():
                    plot_display = gr.Plot(label="评分趋势图")
                    refresh_plot_btn = gr.Button("📊 更新图表", variant="secondary")
        
        # 数据管理标签页
        with gr.Tab("💾 数据管理"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 数据导出")
                    export_btn = gr.Button("📥 导出数据", variant="primary")
                    export_result = gr.Textbox(label="导出结果", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 数据重置")
                    gr.Markdown("⚠️ 此操作将删除所有评分和历史记录")
                    reset_btn = gr.Button("🗑️ 重置所有数据", variant="stop")
                    reset_result = gr.Textbox(label="重置结果", interactive=False)
        
        # 系统配置标签页
        with gr.Tab("⚙️ 系统配置"):
            with gr.Column():
                gr.Markdown("### 文件夹设置")
                config_a_dir = gr.Textbox(value=elo_system.config['a_dir'], label="模型A图片文件夹")
                config_b_dir = gr.Textbox(value=elo_system.config['b_dir'], label="模型B图片文件夹")
                config_caption_dir = gr.Textbox(value=elo_system.config['caption_dir'], label="标题文件夹")
                
                gr.Markdown("### ELO参数")
                config_k_factor = gr.Slider(1, 100, value=elo_system.config['k_factor'], label="K因子 (影响评分变化幅度)")
                config_initial_rating = gr.Slider(1000, 2000, value=elo_system.config['initial_rating'], label="初始评分")
                
                config_save_btn = gr.Button("💾 保存配置", variant="primary")
                config_result = gr.Textbox(label="配置结果", interactive=False)
    
    # 事件绑定
    start_session_btn.click(
        start_new_session,
        outputs=[img_a, img_b, caption_text, result]
    )
    
    btn_a.click(
        lambda: vote("A"),
        outputs=[img_a, img_b, caption_text, result]
    )
    
    btn_b.click(
        lambda: vote("B"),
        outputs=[img_a, img_b, caption_text, result]
    )
    
    skip_btn.click(
        get_current_pair,
        outputs=[img_a, img_b, caption_text, result]
    )
    
    refresh_stats_btn.click(
        get_statistics_display,
        outputs=[stats_display]
    )
    
    refresh_plot_btn.click(
        lambda: create_rating_plot(elo_system),
        outputs=[plot_display]
    )
    
    export_btn.click(
        export_data,
        outputs=[export_result]
    )
    
    reset_btn.click(
        reset_all_data,
        outputs=[reset_result]
    )
    
    config_save_btn.click(
        update_config,
        inputs=[config_a_dir, config_b_dir, config_caption_dir, config_k_factor, config_initial_rating],
        outputs=[config_result]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )