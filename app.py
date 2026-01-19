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