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