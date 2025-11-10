import pandas as pd
import numpy as np
import seaborn as sns

# Force non-GUI backend to avoid Tkinter/thread errors in Flask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Data Loading & Cleaning
def load_and_clean_data(file_path):
    """Load and clean real dataset only."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded dataset with {len(df)} rows")
    
        # Derive Skill_Count from Top_Skills
        if 'Top_Skills' in df.columns:
            df["Skill_Count"] = df["Top_Skills"].fillna("").apply(
                lambda x: len([s.strip() for s in str(x).replace(",", ";").split(";") if s.strip()])
            )
        else:
            df["Skill_Count"] = 0
    
        # Drop incomplete rows
        df = df.dropna(subset=["Year", "Average_Salary", "Sector", "Field"])
    
        # Convert numeric columns
        numeric_cols = ["Year", "Average_Salary", "Demand_Index", "Job_Satisfaction",
                        "Remote_Potential", "Automation_Risk", "Skill_Count"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    
        print(f"✅ Cleaned dataset: {df['Sector'].nunique()} sectors, {df['Field'].nunique()} fields")
        return df


    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Return sample data for demonstration
        return create_sample_data()

def create_sample_data():
    """Create comprehensive sample data for testing"""
    print("📊 Creating sample data for demonstration...")
    
    sectors_fields = {
        "Technology": [
            "Software Development", "Data Science", "Cybersecurity", 
            "Cloud Computing", "AI/ML Engineering", "DevOps"
        ],
        "Healthcare": [
            "Medicine", "Nursing", "Medical Research", 
            "Pharmacy", "Healthcare Administration"
        ],
        "Finance": [
            "Investment Banking", "Financial Analysis", "Accounting",
            "Risk Management", "Wealth Management"
        ]
    }
    
    sample_data = []
    current_id = 0
    
    for sector, fields in sectors_fields.items():
        for field in fields:
            base_salary = np.random.uniform(8, 25)  # Base LPA
            for year in range(2020, 2025):
                current_id += 1
                growth_factor = 1 + (year - 2020) * 0.08  # 8% annual growth
            # Fields     
                sample_data.append({
                    'Year': year,
                    'Sector': sector,
                    'Field': field,
                    'Average_Salary': round(base_salary * growth_factor + np.random.uniform(-1, 1), 2),
                    'Demand_Index': np.random.randint(60, 95),
                    'Job_Satisfaction': np.random.randint(70, 90),
                    'Remote_Potential': np.random.randint(40, 90),
                    'Automation_Risk': np.random.randint(10, 40),
                    'Innovation_Index': np.random.randint(65, 95),
                    'Sustainability_Score': np.random.randint(60, 85),
                    'Top_Skills': 'Python;JavaScript;SQL;Communication;Problem Solving',
                    'Skill_Count': np.random.randint(4, 8)
                })
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Sample data created with {len(df)} rows")
    return df

# Skills Extraction
def extract_skills_from_dataset(df, sector, field):
    """Extract top skills only from dataset for the given field."""
    try:
        field_data = df[(df['Sector'] == sector) & (df['Field'] == field)]

        if field_data.empty or 'Top_Skills' not in df.columns:
            print(f"⚠️ No skills found for {field} in {sector}.")
            return []

        latest_year = field_data['Year'].max()
        latest_data = field_data[field_data['Year'] == latest_year]

        if latest_data.empty:
            latest_data = field_data  # fallback to all years

        # Extract unique skills from Top_Skills column
        all_skills = []
        for skill_str in latest_data['Top_Skills'].dropna():
            skill_str = str(skill_str).replace(",", ";")
            all_skills.extend([s.strip().title() for s in skill_str.split(";") if s.strip()])

        # Deduplicate and clean
        skills = sorted(set(all_skills))
        print(f"✅ Extracted {len(skills)} unique dataset skills for {field}.")
        return skills

    except Exception as e:
        print(f"❌ Skill extraction error: {e}")
        return []



# Salary Match
def calculate_salary_match(expected_salary, predicted_salaries, current_salary):
    """Calculate how well expected salary matches predictions with enhanced logic"""
    if not predicted_salaries or expected_salary <= 0:
        return 50  # Default neutral score
    
    # Use median of predictions
    median_pred = float(np.median(predicted_salaries))
    if median_pred <= 0:
        return 50
    
    # Calculate match percentage (0-100)
    if expected_salary <= median_pred:
        # If expected is less than or equal to predicted, good match
        match_percentage = 80 + (expected_salary / median_pred) * 20
    else:
        # If expected is higher than predicted, penalize but not too harshly
        match_percentage = (median_pred / expected_salary) * 80
    
    return round(min(max(match_percentage, 0), 100), 1)

# Enhanced Forecasting
def forecast_salary(field_data):
    """
    Predict salary trends for the next 3 years using enhanced regression
    with better error handling and realistic projections.
    """
    try:
        if field_data is None or len(field_data) < 2:
            return create_default_forecast()
        
        # Ensure we have enough data points
        if len(field_data) < 2:
            # Duplicate data to have enough points for regression
            field_data = pd.concat([field_data, field_data], ignore_index=True)
        
        # Compute average skill count for that field
        avg_skill_count = field_data["Skill_Count"].mean() if "Skill_Count" in field_data.columns else 5
        
        # Enhanced Skill-Based Scaling with bounds
        skill_influence = 0.03  # 3% per skill difference from field average
        
        if "Skill_Count" in field_data.columns and "Average_Salary" in field_data.columns:
            field_data = field_data.copy()
            field_data["Adjusted_Salary"] = field_data.apply(
                lambda r: r["Average_Salary"] * (1 + min(max((r["Skill_Count"] - avg_skill_count) * skill_influence, -0.2), 0.2)),
                axis=1
            )
            y_col = "Adjusted_Salary"
        else:
            y_col = "Average_Salary"

        # Prepare data for regression
        X = field_data["Year"].values.reshape(-1, 1)
        y = field_data[y_col].values
        
        # Handle cases with insufficient variance
        if len(np.unique(y)) < 2:
            y = y + np.random.normal(0, 0.1, len(y))  # Add small noise
        
        # Train Regression Model with regularization
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # Check model quality
            if model.coef_[0] < 0:  # Negative trend, force minimal positive
                model.coef_[0] = 0.5
        except:
            # Fallback: simple average growth
            return create_simple_forecast(field_data)

        latest_year = int(field_data["Year"].max())
        future_years = np.array([latest_year + i for i in range(1, 4)]).reshape(-1, 1)

        # Base regression predictions with bounds
        predicted = model.predict(future_years)
        latest_salary = float(y[-1]) if len(y) > 0 else field_data["Average_Salary"].mean()
        
        # Ensure predictions are realistic
        predicted = np.maximum(predicted, latest_salary * 0.8)  # At least 80% of current
        predicted = np.minimum(predicted, latest_salary * 1.5)  # At most 150% of current
        
        # Apply skill-based boost with realistic bounds
        field_skill_boost = 1 + min(max((avg_skill_count - 5) * 0.02, -0.1), 0.15)
        
        # Apply growth constraints (max 12% per year)
        growth_cap = 0.12
        adjusted_forecast = []
        for i, base in enumerate(predicted):
            if i == 0:
                max_allowed = latest_salary * (1 + growth_cap)
            else:
                max_allowed = adjusted_forecast[-1] * (1 + growth_cap)
            
            adjusted = min(base * field_skill_boost, max_allowed)
            adjusted_forecast.append(max(adjusted, latest_salary * 0.9))  # At least 90% of current
        
        # Build comprehensive output
        current_stats = get_field_stats(field_data)
        
        return {
            "years": future_years.flatten().astype(int).tolist(),
            "predicted_salary": [float(p) for p in adjusted_forecast],
            "latest_salary": float(latest_salary),
            "current_salary": current_stats['current_salary'],
            "predicted_min": [float(p * 0.85) for p in adjusted_forecast],  # 15% lower bound
            "predicted_max": [float(p * 1.15) for p in adjusted_forecast],  # 15% upper bound
            "avg_skill_count": round(avg_skill_count, 1),
            "skill_boost_factor": round(field_skill_boost, 3),
            "skill_influence": f"{int(skill_influence * 100)}% per skill diff",
            "growth_rate": f"{((adjusted_forecast[-1] / latest_salary - 1) * 100):.1f}% overall",
            "confidence": "High" if len(field_data) >= 3 else "Medium"
        }

    except Exception as e:
        print(f"⚠️ Forecast error: {e}")
        return create_default_forecast()

def create_default_forecast():
    """Create a reasonable default forecast when data is insufficient"""
    return {
        "years": [2025, 2026, 2027],
        "predicted_salary": [12.0, 13.0, 14.0],
        "latest_salary": 11.0,
        "current_salary": 11.0,
        "predicted_min": [10.0, 11.0, 12.0],
        "predicted_max": [14.0, 15.0, 16.0],
        "avg_skill_count": 5.0,
        "skill_boost_factor": 1.0,
        "skill_influence": "3% per skill diff",
        "growth_rate": "10.0% overall",
        "confidence": "Low"
    }

def create_simple_forecast(field_data):
    """Create forecast using simple growth assumptions"""
    current_salary = field_data["Average_Salary"].iloc[-1] if len(field_data) > 0 else 10.0
    growth_rate = 0.08  # 8% annual growth
    
    forecast = {
        "years": [2025, 2026, 2027],
        "predicted_salary": [
            round(current_salary * (1 + growth_rate), 2),
            round(current_salary * (1 + growth_rate)**2, 2),
            round(current_salary * (1 + growth_rate)**3, 2)
        ],
        "latest_salary": round(current_salary, 2),
        "current_salary": round(current_salary, 2),
        "predicted_min": [
            round(current_salary * (1 + growth_rate * 0.7), 2),
            round(current_salary * (1 + growth_rate * 0.7)**2, 2),
            round(current_salary * (1 + growth_rate * 0.7)**3, 2)
        ],
        "predicted_max": [
            round(current_salary * (1 + growth_rate * 1.3), 2),
            round(current_salary * (1 + growth_rate * 1.3)**2, 2),
            round(current_salary * (1 + growth_rate * 1.3)**3, 2)
        ],
        "avg_skill_count": 5.0,
        "skill_boost_factor": 1.0,
        "skill_influence": "3% per skill diff",
        "growth_rate": f"{(growth_rate * 100):.1f}% annual",
        "confidence": "Medium"
    }
    return forecast

# Field Stats
def get_field_stats(field_data):
    """Get comprehensive statistics for a field"""
    if field_data is None or field_data.empty:
        return create_default_field_stats()
    
    try:
        latest_year = field_data['Year'].max()
        latest_data = field_data[field_data['Year'] == latest_year]
        
        if latest_data.empty:
            return create_default_field_stats()
        
        current_row = latest_data.iloc[0]
        
        # Calculate trends
        if len(field_data) > 1:
            prev_year_data = field_data[field_data['Year'] == latest_year - 1]
            salary_trend = "↑ Growing" if len(prev_year_data) > 0 and current_row['Average_Salary'] > prev_year_data['Average_Salary'].iloc[0] else "→ Stable"
            demand_trend = "↑ High" if current_row['Demand_Index'] > 75 else "→ Moderate"
        else:
            salary_trend = "→ Stable"
            demand_trend = "→ Moderate"
        
        return {
            'current_salary': round(float(current_row['Average_Salary']), 2),
            'current_demand': round(float(current_row.get('Demand_Index', 65)), 1),
            'job_satisfaction': round(float(current_row.get('Job_Satisfaction', 70)), 1),
            'remote_potential': round(float(current_row.get('Remote_Potential', 50)), 1),
            'automation_risk': round(float(current_row.get('Automation_Risk', 30)), 1),
            'innovation_index': round(float(current_row.get('Innovation_Index', 70)), 1),
            'sustainability_score': round(float(current_row.get('Sustainability_Score', 65)), 1),
            'salary_trend': salary_trend,
            'demand_trend': demand_trend,
            'data_points': len(field_data)
        }
    except Exception as e:
        print(f"⚠️ Error getting field stats: {e}")
        return create_default_field_stats()

def create_default_field_stats():
    """Create default field statistics"""
    return {
        'current_salary': 12.0,
        'current_demand': 70.0,
        'job_satisfaction': 75.0,
        'remote_potential': 60.0,
        'automation_risk': 25.0,
        'innovation_index': 70.0,
        'sustainability_score': 65.0,
        'salary_trend': "→ Stable",
        'demand_trend': "→ Moderate",
        'data_points': 0
    }

# Enhanced Skill Analysis
def analyze_skills(df, sector, field, expected_salary, user_skills):
    """Comprehensive skill analysis with enhanced matching"""
    try:
        dataset_skills = extract_skills_from_dataset(df, sector, field)
        
        # Normalize skills for comparison
        user_norm = [s.strip().title() for s in user_skills if s and isinstance(s, str)]
        req_norm = [s.strip().title() for s in dataset_skills if s and isinstance(s, str)]
        
        # Enhanced case-insensitive matching with partial matching
        matched = []
        missing = []
        partially_matched = []
        
        for required_skill in req_norm:
            found = False
            partial_match = False
            
            for user_skill in user_norm:
                if user_skill.lower() == required_skill.lower():
                    matched.append(required_skill)
                    found = True
                    break
                elif required_skill.lower() in user_skill.lower() or user_skill.lower() in required_skill.lower():
                    partially_matched.append(f"{user_skill} → {required_skill}")
                    partial_match = True
            
            if not found and not partial_match:
                missing.append(required_skill)
        
        # Calculate match percentage
        match_percent = (len(matched) / len(req_norm) * 100) if req_norm else 0
        
        # Get field data and forecasts
        field_data = df[(df['Sector'] == sector) & (df['Field'] == field)]
        forecast = forecast_salary(field_data)
        field_stats = get_field_stats(field_data)
        
        # Enhanced salary match calculation
        salary_match = calculate_salary_match(
            expected_salary, 
            forecast.get('predicted_salary', []), 
            field_stats['current_salary']
        )
        
        # Generate comprehensive narrative
        narrative = generate_narrative(sector, field, salary_match, match_percent, 
                                     missing, partially_matched, forecast, field_data)
        
        # Calculate overall career score
        # Calculate overall career score
        career_score = calculate_career_score(salary_match, match_percent, field_stats)
        
        # Determine growth potential based on forecast
# Determine growth potential based on forecast
        growth_rate_str = str(forecast.get('growth_rate', '0')).split('%')[0]
        try:
            growth_rate = float(growth_rate_str)
        except ValueError:
            growth_rate = 0.0

        if growth_rate >= 15:
            growth_potential = "High"
        elif growth_rate >= 8:
            growth_potential = "Medium"
        else:
            growth_potential = "Low"
        
        # Salary comparison
        current_salary = field_stats['current_salary']
        if expected_salary > current_salary * 1.2:
            salary_comparison = "above"
            salary_advice = "Your salary expectations are ambitious. Consider gaining additional experience or skills to justify this level."
        elif expected_salary < current_salary * 0.8:
            salary_comparison = "below"
            salary_advice = "Your salary expectations are conservative. You may be undervaluing your potential in this field."
        else:
            salary_comparison = "aligned"
            salary_advice = "Your salary expectations are well-aligned with current market rates."
        
        # Match description based on career score
        if career_score >= 80:
            match_description = "Excellent alignment with this career path! You have strong potential for success."
        elif career_score >= 65:
            match_description = "Good alignment with opportunities for growth and development."
        else:
            match_description = "Consider focusing on skill development and market research to improve your alignment."
        
        # Generate recommendations
        recommendations = []
        automation_risk = field_stats.get('automation_risk', 30)

        if match_percent < 70:
            recommendations.append(f"Focus on developing these key skills: {', '.join(missing[:3])}")
        if salary_match < 70:
            recommendations.append("Research current market rates and adjust your salary expectations accordingly")
        if field_stats['current_demand'] > 75:
            recommendations.append("High market demand - excellent time to pursue opportunities in this field")
        if automation_risk > 50:
            recommendations.append("Focus on developing skills that are automation-resistant")

        
        # Add general recommendations
        recommendations.extend([
            "Network with professionals in this field to gain industry insights",
            "Consider relevant certifications to enhance your credentials",
            "Stay updated with industry trends and emerging technologies"
        ])
        
        # Create the final result structure
        result = {
            # Direct fields for template
            'sector': sector,
            'field': field,
            'match_score': round(career_score, 1),
            'expected_salary': expected_salary,
            'growth_potential': growth_potential,
            'average_salary': round(current_salary, 2),
            'salary_comparison': salary_comparison,
            'salary_advice': salary_advice,
            'match_description': match_description,
            'skills_advice': f"Focus on acquiring {len(missing)} key skills to improve your competitiveness.",
            'growth_description': f"Projected growth rate: {forecast.get('growth_rate', 'N/A')}",
            'timeline': "6-12 months for skill development, 1-2 years for significant career advancement",
            'recommendations': recommendations[:5],
            'matched_skills': matched,
            'missing_skills': missing,
            'required_skills': req_norm,
            "skills": {   # 🟣 Add this
                "total_matched": len(matched),
                "missing": len(missing)
            },

            'required_skills': req_norm,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            
            # Original structure for internal use
            'alignment': {
                'salary_match_percent': salary_match,
                'predicted_salary_min': forecast.get('predicted_min', [0, 0, 0]),
                'predicted_salary_max': forecast.get('predicted_max', [0, 0, 0]),
                'career_score': career_score
            },

            'narrative': narrative,
            'field_stats': field_stats,
            'forecast': forecast,
            'inputs': {
                'sector': sector,
                'field': field,
                'expected_salary': expected_salary
            },
            'analysis_metadata': {
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                'data_quality': 'High' if len(field_data) >= 3 else 'Medium',
                'skills_source': 'Dataset' if dataset_skills else 'Predefined'
            }
        }
        if not result["required_skills"]:
            result["skills_advice"] = "No skill data available in dataset for this field. Consider enriching dataset for better insights."


        return result
        
    except Exception as e:
        print(f"❌ Error in skill analysis: {e}")
        return create_error_result(sector, field, expected_salary, str(e))

def calculate_career_score(salary_match, skills_match, field_stats):
    """Calculate overall career compatibility score"""
    # Weighted average of different factors
    salary_weight = 0.4
    skills_weight = 0.3
    demand_weight = 0.2
    satisfaction_weight = 0.1
    
    demand_score = min(field_stats['current_demand'] / 100 * 100, 100)
    satisfaction_score = field_stats['job_satisfaction']
    
    overall_score = (
        salary_match * salary_weight +
        skills_match * skills_weight +
        demand_score * demand_weight +
        satisfaction_score * satisfaction_weight
    )
    
    return round(overall_score, 1)

def create_error_result(sector, field, expected_salary, error_msg):
    """Create a result structure for error cases"""
    return {
        'alignment': {
            'salary_match_percent': 0,
            'predicted_salary_min': [0, 0, 0],
            'predicted_salary_max': [0, 0, 0],
            'career_score': 0
        },
        'skills': {
            'match_percent': 0,
            'missing_skills': [],
            'required_skills': [],
            'matched_skills': [],
            'partially_matched': [],
            'total_required': 0,
            'total_matched': 0
        },
        'narrative': f"Analysis encountered an error: {error_msg}. Please try again with different parameters.",
        'field_stats': create_default_field_stats(),
        'forecast': create_default_forecast(),
        'inputs': {
            'sector': sector,
            'field': field,
            'expected_salary': expected_salary
        },
        'analysis_metadata': {
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            'data_quality': 'Low',
            'skills_source': 'Error'
        }
    }

# Enhanced Narrative Generation
def generate_narrative(sector, field, salary_match, match_percent, missing_skills, partially_matched, forecast, df_filtered):
    """Generate comprehensive career narrative"""
    narrative = []
    
    # Career Score Summary
    career_score = calculate_career_score(salary_match, match_percent, get_field_stats(df_filtered))
    narrative.append(f"🏆 **Overall Career Compatibility: {career_score}/100**")
    
    # Salary alignment
    if salary_match >= 90:
        narrative.append(f"💵 **Excellent salary alignment** ({salary_match}%) - Your expectations are well-matched with market rates.")
    elif salary_match >= 75:
        narrative.append(f"💰 **Good salary alignment** ({salary_match}%) - Your expectations are reasonable for this field.")
    elif salary_match >= 60:
        narrative.append(f"📊 **Moderate salary alignment** ({salary_match}%) - Consider adjusting expectations or upskilling.")
    else:
        narrative.append(f"⚡ **Salary alignment needs improvement** ({salary_match}%) - Significant gap between expectations and market rates.")
    
    # Skills match
    if match_percent >= 85:
        narrative.append("🎯 **Exceptional skills match** - You have most required competencies!")
    elif match_percent >= 65:
        narrative.append("✅ **Good skills foundation** - You have many key skills with some development areas.")
    elif match_percent >= 40:
        narrative.append("📚 **Moderate skills match** - Focus on developing core competencies.")
    else:
        narrative.append("🚀 **Significant upskilling opportunity** - Consider focused skill development.")
    
    # Missing skills focus
    if missing_skills:
        narrative.append(f"🎯 **Priority Skills to Develop**: {', '.join(missing_skills[:3])}")
    
    if partially_matched:
        narrative.append(f"🔄 **Skill Gaps to Bridge**: {', '.join(partially_matched[:2])}")
    
    # Market insights
    field_stats = get_field_stats(df_filtered)
    
    # Demand analysis
    if field_stats['current_demand'] >= 80:
        narrative.append("📈 **High Market Demand** - Excellent job opportunities available.")
    elif field_stats['current_demand'] >= 65:
        narrative.append("↗️ **Growing Demand** - Good prospects with competitive hiring.")
    else:
        narrative.append("📋 **Stable Market** - Consistent opportunities with focused search.")
    
    # Automation risk
    automation_risk = field_stats['automation_risk']
    if automation_risk < 20:
        narrative.append("🛡️ **Low Automation Risk** - High job security in this field.")
    elif automation_risk < 40:
        narrative.append("✅ **Moderate-Low Automation Risk** - Good long-term stability.")
    elif automation_risk < 60:
        narrative.append("⚠️ **Moderate Automation Risk** - Stay updated with emerging technologies.")
    else:
        narrative.append("🔔 **Higher Automation Risk** - Focus on uniquely human skills.")
    
    # Remote work potential
    remote_potential = field_stats['remote_potential']
    if remote_potential >= 75:
        narrative.append("🏠 **High Remote Work Potential** - Excellent flexibility options.")
    elif remote_potential >= 50:
        narrative.append("💻 **Moderate Remote Potential** - Hybrid opportunities available.")
    
    # Growth forecast
    growth_rate = forecast.get('growth_rate', '0%')
    narrative.append(f"📊 **Projected Growth**: {growth_rate} over 3 years")
    
    return " ".join(narrative)

# Enhanced Chart Generation
def generate_charts(df, sector, field, forecast, chart_dir="static/charts"):
    """Generate main visualizations for salary and performance."""
    os.makedirs(chart_dir, exist_ok=True)
    df_filtered = df[(df['Sector'] == sector) & (df['Field'] == field)]
    charts = {}
    base = f"{sector}_{field}".replace(" ", "_")

    # 1️⃣ Salary Trend Chart
    plt.figure(figsize=(10,6))
    plt.plot(df_filtered["Year"], df_filtered["Average_Salary"], "o-", label="Historical", color="#6366F1")
    if forecast and forecast.get("years"):
        plt.plot(forecast["years"], forecast["predicted_salary"], "s--", label="Forecast", color="#10B981")
    plt.title(f"Salary Trend — {field}")
    plt.xlabel("Year")
    plt.ylabel("Salary (LPA)")
    plt.legend()
    path1 = os.path.join(chart_dir, f"{base}_trend.png")
    plt.savefig(path1, bbox_inches="tight")
    charts["trend"] = f"static/charts/{os.path.basename(path1)}"
    plt.close()

    # 2️⃣ Skill Count vs Salary
    plt.figure(figsize=(8,6))
    plt.scatter(df_filtered["Skill_Count"], df_filtered["Average_Salary"], color="#8B5CF6", alpha=0.7)
    plt.title("Skill Impact on Salary")
    plt.xlabel("Number of Skills")
    plt.ylabel("Average Salary (LPA)")
    z = np.polyfit(df_filtered["Skill_Count"], df_filtered["Average_Salary"], 1)
    plt.plot(sorted(df_filtered["Skill_Count"].unique()), np.poly1d(z)(sorted(df_filtered["Skill_Count"].unique())), color="#10B981", linestyle="--")
    path2 = os.path.join(chart_dir, f"{base}_skill_impact.png")
    plt.savefig(path2, bbox_inches="tight")
    charts["skill_impact"] = f"static/charts/{os.path.basename(path2)}"
    plt.close()

    # 3️⃣ Demand vs Automation Risk
    plt.figure(figsize=(8,6))
    plt.scatter(df_filtered["Demand_Index"], df_filtered["Automation_Risk"], color="#F59E0B", alpha=0.7)
    plt.title("Demand vs Automation Risk")
    plt.xlabel("Demand Index")
    plt.ylabel("Automation Risk")
    path3 = os.path.join(chart_dir, f"{base}_demand_automation.png")
    plt.savefig(path3, bbox_inches="tight")
    charts["demand_automation"] = f"static/charts/{os.path.basename(path3)}"
    plt.close()

    # 4️⃣ Job Satisfaction & Remote Potential
    plt.figure(figsize=(8,6))
    plt.bar(df_filtered["Year"], df_filtered["Job_Satisfaction"], alpha=0.7, label="Job Satisfaction", color="#6366F1")
    plt.plot(df_filtered["Year"], df_filtered["Remote_Potential"], "o--", color="#10B981", label="Remote Potential")
    plt.title("Job Satisfaction & Remote Work Trend")
    plt.xlabel("Year")
    plt.legend()
    path4 = os.path.join(chart_dir, f"{base}_satisfaction_remote.png")
    plt.savefig(path4, bbox_inches="tight")
    charts["satisfaction_remote"] = f"static/charts/{os.path.basename(path4)}"
    plt.close()

    # 5️⃣ Salary Distribution
    plt.figure(figsize=(8,6))
    sns.histplot(df_filtered["Average_Salary"], bins=10, color="#4F46E5", kde=True)
    plt.title("Salary Distribution")
    plt.xlabel("Average Salary (LPA)")
    plt.ylabel("Frequency")
    path5 = os.path.join(chart_dir, f"{base}_salary_dist.png")
    plt.savefig(path5, bbox_inches="tight")
    charts["salary_dist"] = f"static/charts/{os.path.basename(path5)}"
    plt.close()

    return charts


# Main Analysis Function
def analyze_career(df, sector, field, expected_salary, user_skills, name="User"):
    """Main function to analyze career data and generate insights."""
    print(f"🔍 Starting analysis for {sector} - {field}")
    
    # Get field data
    field_data = df[(df['Sector'] == sector) & (df['Field'] == field)]
    
    if field_data.empty:
        error_msg = f"No data found for {field} in {sector} sector."
        print(f"❌ {error_msg}")
        return {'error': error_msg}
    
    print(f"✅ Found {len(field_data)} data points for analysis")
    
    # Generate forecasts and analysis
    forecast = forecast_salary(field_data)
    analysis = analyze_skills(df, sector, field, expected_salary, user_skills)
    
    # Generate charts
    print("📊 Generating charts...")
    main_charts = generate_charts(df, sector, field, forecast)
    
    # Combine all results
    result = {
        **analysis,
        'charts': {**main_charts, },
        'forecast': forecast,
        'name': name
    }
    result["skills"] = {
        "total_matched": len(result.get("matched_skills", [])),
        "missing": len(result.get("missing_skills", [])),
        "total_required": len(result.get("required_skills", []))
    }
    charts = result.get("charts", {})
    result["charts_display"] = {
        "trend": charts.get("trend"),
        "skill_impact": charts.get("skill_impact"),
        "demand": charts.get("demand_automation"),
        "satisfaction": charts.get("satisfaction_remote"),
        "salary_dist": charts.get("salary_dist")
    }
    
    print(f"✅ Analysis complete for {name}")
    return result