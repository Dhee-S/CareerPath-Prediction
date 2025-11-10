from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import numpy as np
from analysis import (
    analyze_career,
    load_and_clean_data,
    generate_charts,
    analyze_skills
)
import os
from datetime import timedelta
import traceback
import json

app = Flask(__name__)
app.config['CHART_DIR'] = 'static/charts'
app.secret_key = 'career-analyzer-secret-key-2024'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Load data once at startup
df = None
try:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'career_data.csv')
    print(f"Looking for CSV at: {csv_path}")
    print(f"File exists: {os.path.exists(csv_path)}")
    
    if os.path.exists(csv_path):
        df = load_and_clean_data(csv_path)
        print("Data loaded successfully")
        print(f"Available sectors: {df['Sector'].unique().tolist()}")
    else:
        print("CSV file not found at the specified path")
        print("Files in current directory:", os.listdir('.'))
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()

def get_available_options():
    """Get all available sectors, fields, and their skills from dataset"""
    if df is None:
        return {'sectors': [], 'mapping': {}, 'field_skills': {}}
    
    sectors = sorted(df['Sector'].dropna().unique().tolist())
    mapping = {}
    field_skills = {}
    
    for sector in sectors:
        sector_fields = sorted(df[df['Sector'] == sector]['Field'].dropna().unique().tolist())
        mapping[sector] = sector_fields
        
        # Get skills for each field in this sector (from most recent year only)
        for field in sector_fields:
            field_data = df[(df['Sector'] == sector) & (df['Field'] == field)]
            if not field_data.empty:
                # Get the most recent year's data
                latest_year = field_data['Year'].max()
                latest_data = field_data[field_data['Year'] == latest_year]
                
                if not latest_data.empty and 'Top_Skills' in latest_data.columns:
                    skills_str = latest_data['Top_Skills'].iloc[0]
                    if pd.notna(skills_str):
                        # Handle both semicolon and comma separated skills
                        if ';' in skills_str:
                            skills_list = [skill.strip() for skill in str(skills_str).split(';') if skill.strip()]
                        elif ',' in skills_str:
                            skills_list = [skill.strip() for skill in str(skills_str).split(',') if skill.strip()]
                        else:
                            skills_list = [skills_str.strip()]
                        field_skills[f"{sector}||{field}"] = skills_list
    
    return {
        'sectors': sectors,
        'mapping': mapping,
        'field_skills': field_skills
    }

@app.route('/')
def index():
    """Main page with input form"""
    if df is None:
        return render_template('error.html', message="Data not available. Please check the CSV file.")
    
    options = get_available_options()
    
    # Get last session data for pre-filling
    last_inputs = session.get('last_analysis', {})
    
    return render_template('index.html',
                         sectors=options['sectors'],
                         mapping=options['mapping'],
                         last_inputs=last_inputs)

@app.route('/api/options')
def api_options():
    """API endpoint to get available sectors and fields"""
    options = get_available_options()
    return jsonify(options)

@app.route('/api/field-skills/<sector>/<field>')
def api_field_skills(sector, field):
    """API endpoint to get skills for a specific field"""
    options = get_available_options()
    field_key = f"{sector}||{field}"
    
    field_specific_skills = options['field_skills'].get(field_key, [])
    
    return jsonify({
        'field_skills': field_specific_skills
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for career analysis"""
    if df is None:
        return jsonify({'error': 'Data not available'}), 500

    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['sector', 'field', 'expected_salary']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Optional values
        name = data.get('name', 'User')
        skills = data.get('skills', [])
        expected_salary = float(data['expected_salary'])

        # 🔹 Perform career analysis (uses your main analysis function)
        result = analyze_career(df, data['sector'], data['field'], expected_salary, skills, name)

        # 🔹 Store the result in session so /result can access it
        session['analysis_result'] = result
        session['last_inputs'] = {
            'name': name,
            'sector': data['sector'],
            'field': data['field'],
            'expected_salary': expected_salary
        }

        # ✅ Return success JSON (frontend can redirect to /result)
        return jsonify({'success': True, 'redirect': '/result'})

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Web form analysis endpoint"""
    if df is None:
        return render_template('error.html', message="Data not available.")

    try:
        name = request.form.get("name", "User")
        sector = request.form.get("sector")
        field = request.form.get("field")
        expected_salary = float(request.form.get("expected_salary", 0))
        user_skills = request.form.getlist("skills")  # handles multiple checkboxes

        # 🔹 Analyze the data
        result = analyze_career(df, sector, field, expected_salary, user_skills, name)

        # 🔹 Save to session
        session["analysis_result"] = result
        session["last_inputs"] = {
            "name": name,
            "sector": sector,
            "field": field,
            "expected_salary": expected_salary
        }

        # ✅ Render directly without redirect, or redirect if preferred
        return render_template("result.html", result=result, name=name)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html', message=f"Analysis error: {str(e)}")

    
@app.route('/result')
def result():
    """Result page endpoint"""
    result_data = session.get('analysis_result')
    if not result_data:
        return render_template('error.html', message="No analysis results found. Please complete the analysis first.")
    name = session.get('last_inputs', {}).get('name', 'User')
    return render_template('result.html', result=result_data, name=name)


@app.route('/api/session/restore')
def restore_session():
    """API endpoint to restore previous session"""
    if 'last_analysis' in session:
        return jsonify(session['last_analysis'])
    return jsonify({'error': 'No session found'}), 404

if __name__ == '__main__':
    # Create chart directory if it doesn't exist
    os.makedirs(app.config['CHART_DIR'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)