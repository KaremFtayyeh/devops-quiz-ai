from flask import Flask, request, jsonify, send_from_directory
import db, ai, os

app = Flask(__name__, static_folder='static')

# ── Static files ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ── Health check ──────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

# ── Student info (powers the footer) ─────────────────────────────────────────
@app.route('/api/info')
def info():
    return jsonify({'student_id': os.environ.get('STUDENT_ID', 'NOT SET')})

# ── Session start ─────────────────────────────────────────────────────────────
@app.route('/api/session/start', methods=['POST'])
def session_start():
    data       = request.get_json()
    topic      = data.get('topic', 'general')
    difficulty = data.get('difficulty', 'medium')
    conn       = db.get_connection()
    cur        = conn.execute(
        'INSERT INTO sessions (topic, difficulty) VALUES (?, ?)',
        (topic, difficulty)
    )
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return jsonify({'session_id': session_id})

# ── Record answer ─────────────────────────────────────────────────────────────
@app.route('/api/session/answer', methods=['POST'])
def session_answer():
    data = request.get_json()
    conn = db.get_connection()
    conn.execute(
        'INSERT INTO answers (session_id, question_id, topic, selected, correct) VALUES (?,?,?,?,?)',
        (data['session_id'], data['question_id'], data['topic'], data['selected'], data['correct'])
    )
    conn.commit()
    conn.close()
    return jsonify({'ok': True})

# ── Generate questions ────────────────────────────────────────────────────────
@app.route('/api/generate', methods=['POST'])
def generate():
    data       = request.get_json()
    topic      = data.get('topic', 'general')
    difficulty = data.get('difficulty', 'medium')
    count      = data.get('count', 5)
    try:
        questions = ai.generate_questions(topic, difficulty, count)
        return jsonify(questions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Explain answer ────────────────────────────────────────────────────────────
@app.route('/api/explain', methods=['POST'])
def explain():
    data = request.get_json()
    try:
        explanation = ai.explain_answer(
            data['question'], data['options'],
            data['answer_index'], data['selected_index']
        )
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'explanation': 'Explanation unavailable.'}), 500

# ── Study recommendation ──────────────────────────────────────────────────────
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data       = request.get_json()
    session_id = data.get('session_id')
    conn       = db.get_connection()
    rows       = conn.execute(
        'SELECT topic, SUM(correct) as correct, COUNT(*) as total '
        'FROM answers WHERE session_id = ? GROUP BY topic',
        (session_id,)
    ).fetchall()
    conn.close()
    topic_scores = [dict(r) for r in rows]
    try:
        rec = ai.recommend_study(topic_scores)
        return jsonify({'recommendation': rec})
    except Exception as e:
        return jsonify({'recommendation': 'Recommendation unavailable.'}), 500

if __name__ == '__main__':
    db.init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
