import nltk
import os
import sqlite3
import traceback
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import config
import json
from flask_socketio import SocketIO, emit
from rag_system import GraphRAGSystem
from vector_db import VectorDBManager
from knowledge_graph import KnowledgeGraphManager
from flask import jsonify

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'doc', 'docx', 'md'}
app.secret_key = 'hospital_secret_key_123'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'md'}

# 在文件开头添加
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['DATABASE'] = os.path.join(BASE_DIR, 'data/hospital.db')
app.config['KNOWLEDGE_BASE'] = config.EXTERNAL_FILE
app.config['VECTOR_DB_PATH'] = os.path.join(BASE_DIR, config.VECTOR_DB_PATH)
app.config['DOCUMENTS_DIR'] = os.path.join(BASE_DIR, config.DOCUMENTS_DIR)

# 确保目录存在
os.makedirs(os.path.dirname(app.config['DATABASE']), exist_ok=True)
os.makedirs(app.config['KNOWLEDGE_BASE'], exist_ok=True)
os.makedirs(app.config['VECTOR_DB_PATH'], exist_ok=True)
os.makedirs(app.config['DOCUMENTS_DIR'], exist_ok=True)

# 修改nltk路径
nltk_data_path = os.path.join(BASE_DIR, 'nltk_data')
nltk.data.path.append(nltk_data_path)

socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化GraphRAG系统
kg_manager = KnowledgeGraphManager()
vdb_manager = VectorDBManager()
graph_rag = GraphRAGSystem(kg_manager, vdb_manager)

# 获取默认配置
@app.route('/admin/system_settings/defaults')
def get_default_config():
    if 'admin_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(config.DEFAULT_CONFIG)


# 在 init_db() 函数中，确保 prompt_templates 表的创建代码正确添加
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()

    # 创建患者表（扩展健康信息字段）
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            blood_type TEXT,
            height TEXT,
            weight TEXT,
            conditions TEXT,
            allergies TEXT,
            occupation TEXT,
            ethnicity TEXT,
            main_activity TEXT,
            education TEXT,
            employment TEXT,
            marital_status TEXT,
            is_smoker TEXT,
            is_drinker TEXT,
            surgery_history TEXT,
            medications TEXT,
            disease_history TEXT,
            systolic_bp TEXT,
            diastolic_bp TEXT,
            bp_measure_time TEXT,
            family_history TEXT,
            regular_exercise TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 检查并添加缺失的列
    columns_to_add = [
        ('occupation', 'TEXT'),
        ('ethnicity', 'TEXT'),
        ('main_activity', 'TEXT'),
        ('education', 'TEXT'),
        ('employment', 'TEXT'),
        ('marital_status', 'TEXT'),
        ('is_smoker', 'TEXT'),
        ('is_drinker', 'TEXT'),
        ('surgery_history', 'TEXT'),
        ('medications', 'TEXT'),
        ('disease_history', 'TEXT'),
        ('systolic_bp', 'TEXT'),
        ('diastolic_bp', 'TEXT'),
        ('bp_measure_time', 'TEXT'),
        ('family_history', 'TEXT'),
        ('regular_exercise', 'TEXT')
    ]

    c.execute("PRAGMA table_info(patients)")
    existing_columns = [col[1] for col in c.fetchall()]

    for column, col_type in columns_to_add:
        if column not in existing_columns:
            c.execute(f"ALTER TABLE patients ADD COLUMN {column} {col_type}")

    # 创建管理员表
    c.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # 创建就诊历史表
    c.execute('''
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            department TEXT NOT NULL,
            doctor TEXT NOT NULL,
            description TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')

    # 创建检查指标表
    c.execute('''
        CREATE TABLE IF NOT EXISTS check_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            item TEXT NOT NULL,
            result TEXT NOT NULL,
            reference_range TEXT NOT NULL,
            unit TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')

    # 创建知识文档表
    c.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,  -- 'file' or 'url'
            path TEXT NOT NULL,   -- 文件路径或URL
            tags TEXT,            -- 逗号分隔的标签
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建 Prompt 模板表 - 确保这个表被创建
    c.execute('''
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            is_active BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 添加默认管理员
    c.execute("SELECT COUNT(*) FROM admins WHERE username = 'admin'")
    if c.fetchone()[0] == 0:
        hashed_password = generate_password_hash('admin123')
        c.execute("INSERT INTO admins (username, password) VALUES (?, ?)",
                  ('admin', hashed_password))

    # 添加默认的 Prompt 模板
    c.execute("SELECT COUNT(*) FROM prompt_templates")
    if c.fetchone()[0] == 0:
        default_templates = [
            ('健康知识推送模板', '用于生成个性化健康知识推送的模板',
             '''你是一名健康知识助手，请基于下面用户的健康画像与知识图谱结果，用专业的语句推送出**个性化健康知识**，要求：

- 仅围绕用户 **真实健康状况** 与 **知识图谱中的有效信息**
- 分点推送出有关系的健康知识，并且要明确标注每个知识点的来源网址
- 要求推送的健康知识与用户的健康状况、知识图谱有效信息强相关
- 可以稍微给出在*饮食、运动、用药、复查、注意事项等具体可操作建议
- 以 Markdown 格式输出，可含小标题、列表、表情符号
- **重要**：在每个知识点后面必须用 [来源](URL) 的格式标注来源链接，URL 必须完整可点击

---
### 👤 用户健康画像
{user_input}

---

### 🔍 知识图谱匹配结果
{kg_results}

---

### 📄 相关文档片段
{vdb_results}

---

请开始生成 **专属健康知识推送**，确保每个知识点都有明确的来源标注：''', 'health_knowledge', 1),

            ('通用问答模板', '适用于一般知识问答场景',
             '''你是一个专业的知识问答助手，请基于以下知识回答用户问题：

### 相关知识：
{kg_results}

### 相关文档：
{vdb_results}

### 用户问题：
{user_input}

请基于以上信息给出专业、准确的回答：''', 'general', 0),

            ('医学诊断建议模板', '用于生成医学诊断建议',
             '''你是一名专业的医学顾问，请基于患者的健康信息和相关医学知识提供诊断建议：

### 患者信息：
{user_input}

### 医学知识图谱：
{kg_results}

### 相关医学文献：
{vdb_results}

请提供专业的医学建议，包括可能的诊断、建议检查和注意事项：''', 'medical', 0)
        ]

        c.executemany('''
            INSERT INTO prompt_templates (name, description, content, category, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', default_templates)

    conn.commit()
    conn.close()

    # 添加测试患者数据
    def add_test_patients():
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()

        patients = [
            ('张伟', '13800138000', generate_password_hash('password123'), 42, '男', 'O型', '175cm', '72kg',
             '轻度高血压',
             '青霉素、花粉'),
            ('李娜', '13900139000', generate_password_hash('abc123'), 35, '女', 'A型', '162cm', '55kg', 'II型糖尿病',
             '无'),
            ('王强', '13700137000', generate_password_hash('pass1234'), 58, '男', 'B型', '178cm', '80kg', '冠心病',
             '海鲜'),
            ('赵敏', '13600136000', generate_password_hash('securepwd'), 29, '女', 'AB型', '168cm', '58kg', '健康',
             '无'),
            ('刘洋', '13500135000', generate_password_hash('mypassword'), 65, '男', 'O型', '170cm', '68kg',
             '慢性支气管炎',
             '花粉、尘螨')
        ]

        try:
            c.executemany('''
                INSERT INTO patients (name, phone, password, age, gender, blood_type, height, weight, conditions, allergies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', patients)
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # 数据已存在

        # 添加测试就诊记录
        for patient_id in range(1, 6):
            records = [
                (patient_id, '2023-10-15', '心血管内科', '王主任', '患者主诉近期偶有头晕现象，血压测量为145/92mmHg'),
                (patient_id, '2023-08-22', '体检中心', '李医生', '年度体检结果显示：血脂略高（LDL 3.5mmol/L）'),
                (patient_id, '2023-06-10', '呼吸科', '张医生', '患者因季节性花粉过敏就诊，症状包括打喷嚏、流涕')
            ]
            c.executemany('''
                INSERT INTO medical_records (patient_id, date, department, doctor, description)
                VALUES (?, ?, ?, ?, ?)
            ''', records)

        # 添加测试检查指标
        for patient_id in range(1, 6):
            metrics = [
                (patient_id, '血压', '142/88', '90-120/60-80', 'mmHg', '2023-10-15', 'warning'),
                (patient_id, '空腹血糖', '5.8', '3.9-6.1', 'mmol/L', '2023-10-15', 'normal'),
                (patient_id, '总胆固醇', '5.3', '<5.2', 'mmol/L', '2023-08-22', 'warning')
            ]
            c.executemany('''
                INSERT INTO check_metrics (patient_id, item, result, reference_range, unit, date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', metrics)

        conn.commit()
        conn.close()