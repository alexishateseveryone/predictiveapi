# bsit_runner.py - Updated to work with new questionnaire structure
import pandas as pd
import pickle
import sys
import json
import os

def debug_print(message):
    """Print debug messages to stderr so they don't interfere with the final output"""
    print(f"DEBUG: {message}", file=sys.stderr)

try:
    debug_print("=== ICT Track Prediction Started ===")
    
    # 1. Load the saved model & encoders
    try:
        model_path = 'rf_ict_model.pkl'
        if not os.path.exists(model_path):
            # Try different possible locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), 'rf_ict_model.pkl'),
                os.path.join('C:\\xampp\\htdocs\\Capstone\\', 'rf_ict_model.pkl')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                rf = model_data['model']
                le_target = model_data['target_encoder']
                feature_names = model_data['feature_names']
                debug_print("✓ Loaded new model format")
            else:
                # Old format fallback
                rf, multi_choice_encoders, le_target = model_data
                feature_names = getattr(rf, 'feature_names_in_', [])
                debug_print("✓ Loaded old model format")
        
        debug_print(f"✓ Available tracks: {le_target.classes_}")
        debug_print(f"✓ Model expects {len(feature_names)} features")
    except Exception as e:
        debug_print(f"✗ Failed to load model: {e}")
        raise

    # 2. Load user data from JSON file
    try:
        user_file = sys.argv[1]
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        debug_print(f"✓ User data loaded from {user_file}")
        debug_print(f"✓ Data contains {len(user_data)} fields")
    except Exception as e:
        debug_print(f"✗ Failed to load user data: {e}")
        raise

    # 3. Debug: Show what data we received
    debug_print("\n=== RECEIVED USER DATA ===")
    for key, value in user_data.items():
        debug_print(f"  '{key}': '{value}'")

    # 4. Rule-based prediction for comparison (basic tracks only)
    def rule_based_predict(data):
        """Rule-based prediction returning only basic tracks (BSIT, BSCS, BSCPE)"""
        scores = {'BSCS': 0, 'BSIT': 0, 'BSCPE': 0}
        
        # Helper to get rating values
        def get_rating(value):
            try:
                return int(value) if str(value).isdigit() else 0
            except:
                return 0
        
        # Calculate scores based on sections
        creative_score = 0
        analytical_score = 0
        networking_score = 0
        creative_count = 0
        analytical_count = 0
        networking_count = 0
        
        # Count creative questions (Section 2)
        for col_name, value in data.items():
            if isinstance(col_name, str) and any(keyword in col_name.lower() for keyword in ['designing', 'editing', 'creating', 'visual', 'graphics', 'animation', 'colors', 'drawing', 'creative']):
                creative_score += get_rating(value)
                creative_count += 1
        
        # Count analytical questions (Section 3)
        for col_name, value in data.items():
            if isinstance(col_name, str) and any(keyword in col_name.lower() for keyword in ['numbers', 'statistics', 'data', 'analytics', 'patterns', 'logical', 'math', 'programming', 'algorithms']):
                analytical_score += get_rating(value)
                analytical_count += 1
        
        # Count networking questions (Section 4)
        for col_name, value in data.items():
            if isinstance(col_name, str) and any(keyword in col_name.lower() for keyword in ['computers', 'connect', 'internet', 'network', 'hardware', 'routers', 'servers', 'technical', 'cables']):
                networking_score += get_rating(value)
                networking_count += 1
        
        # Normalize scores
        if creative_count > 0:
            creative_score = creative_score / creative_count
        if analytical_count > 0:
            analytical_score = analytical_score / analytical_count
        if networking_count > 0:
            networking_score = networking_score / networking_count
        
        debug_print(f"\n=== RULE-BASED ANALYSIS ===")
        debug_print(f"Creative score: {creative_score:.2f} (from {creative_count} questions)")
        debug_print(f"Analytical score: {analytical_score:.2f} (from {analytical_count} questions)")
        debug_print(f"Networking score: {networking_score:.2f} (from {networking_count} questions)")
        
        # Determine main track (only basic 3 tracks)
        # BSCPE: Strong networking preference
        if networking_score >= 3.5 and networking_score > max(creative_score, analytical_score):
            scores['BSCPE'] = networking_score
            scores['BSCS'] = analytical_score * 0.7
            scores['BSIT'] = max(creative_score, analytical_score) * 0.8
        
        # BSCS: Strong analytical preference WITH creative elements (computer science pattern)
        elif analytical_score >= 4.5 and creative_score >= 3.0 and analytical_score > networking_score:
            # Pure CS: High analytical + decent creative (programming + design)
            scores['BSCS'] = analytical_score * 1.1 + (creative_score * 0.4)
            scores['BSIT'] = max(creative_score, analytical_score) * 0.85
            scores['BSCPE'] = networking_score * 0.7
        
        # BSCS fallback: Very high analytical even without creative
        elif analytical_score >= 4.8 and analytical_score > max(creative_score, networking_score) * 1.3:
            scores['BSCS'] = analytical_score * 1.05
            scores['BSIT'] = max(creative_score, analytical_score) * 0.9
            scores['BSCPE'] = networking_score * 0.7
        
        # BSIT: Everything else (creative focus, mixed preferences, or general)
        else:
            scores['BSIT'] = max(creative_score, analytical_score, 3.0)
            scores['BSCS'] = analytical_score * 0.8
            scores['BSCPE'] = networking_score * 0.8
        
        debug_print(f"  BSIT: General/Creative track = {scores['BSIT']:.2f}")
        debug_print(f"  BSCS: Analytical+Creative = {scores['BSCS']:.2f}")
        debug_print(f"  BSCPE: Networking focus = {scores['BSCPE']:.2f}")
        
        debug_print(f"Final rule-based scores: {scores}")
        
        # Return highest scoring track AND the scores + specialization info
        max_score = max(scores.values())
        winner = [k for k, v in scores.items() if v == max_score][0]
        
        # Determine specialization for each track
        specialization = None
        if winner == 'BSIT':
            # BSIT can be Multimedia OR Data Analytics based on preferences
            if creative_score > analytical_score:
                specialization = 'Multimedia'
                debug_print(f"BSIT specialization: Multimedia (creative: {creative_score:.2f} > analytical: {analytical_score:.2f})")
            else:
                specialization = 'Data Analytics'
                debug_print(f"BSIT specialization: Data Analytics (analytical: {analytical_score:.2f} >= creative: {creative_score:.2f})")
        elif winner == 'BSCS':
            # BSCS always gets Data Analytics specialization
            specialization = 'Data Analytics'
            debug_print(f"BSCS specialization: Data Analytics (computer science focus on data)")
        elif winner == 'BSCPE':
            # BSCPE always gets Networking specialization
            specialization = 'Networking'
            debug_print(f"BSCPE specialization: Networking (computer engineering focus on networks)")
        
        debug_print(f"Rule-based winner: {winner} (score: {max_score:.2f})")
        
        return winner, scores, specialization  # Return track, scores, and specialization
    
    # Run rule-based prediction
    rule_prediction, rule_scores, track_specialization = rule_based_predict(user_data)
    
    # 5. Process data for ML model
    debug_print(f"\n=== PROCESSING DATA FOR ML MODEL ===")
    
    df_user = pd.DataFrame([user_data])
    
    # Convert rating columns to numeric
    rating_cols = [col for col in df_user.columns if col not in ['Recommended_Track', 'Timestamp', 'Email Address', 'Full Name', 'Age', 'Gender', 'Strand']]
    debug_print(f"Processing {len(rating_cols)} rating columns...")
    
    for col in rating_cols:
        if col in df_user.columns:
            original = df_user[col].iloc[0] if not df_user[col].empty else 'N/A'
            df_user[col] = pd.to_numeric(df_user[col], errors='coerce').fillna(3)
            new_val = df_user[col].iloc[0]
            debug_print(f"  {col}: '{original}' -> {new_val}")
    
    # 6. Remove columns that shouldn't be features
    columns_to_drop = ['Recommended_Track', 'Timestamp', 'Email Address', 'Full Name', 'Age', 'Gender', 'Strand']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_user.columns]
    if existing_cols_to_drop:
        df_user = df_user.drop(columns=existing_cols_to_drop)
        debug_print(f"Dropped columns: {existing_cols_to_drop}")

    # 7. Convert any remaining object columns to numeric
    for col in df_user.columns:
        if df_user[col].dtype == 'object':
            original = df_user[col].iloc[0] if not df_user[col].empty else 'N/A'
            df_user[col] = pd.to_numeric(df_user[col], errors='coerce').fillna(0)
            new_val = df_user[col].iloc[0]
            debug_print(f"Object->numeric {col}: '{original}' -> {new_val}")

    # 8. Ensure we have all features the model expects
    debug_print(f"Model expects {len(feature_names)} features")
    debug_print(f"We have {len(df_user.columns)} features")
    
    missing_features = []
    for feature in feature_names:
        if feature not in df_user.columns:
            df_user[feature] = 0
            missing_features.append(feature)
    
    if missing_features:
        debug_print(f"Added {len(missing_features)} missing features with value 0")
    
    # Keep only expected features in correct order
    if len(feature_names) > 0:
        df_user = df_user[feature_names]
    debug_print(f"Final feature matrix shape: {df_user.shape}")

    # 9. Make ML prediction
    try:
        ml_pred = rf.predict(df_user)[0]
        ml_track = le_target.inverse_transform([ml_pred])[0]
        
        # Get prediction probabilities for debugging
        ml_proba = rf.predict_proba(df_user)[0]
        proba_dict = dict(zip(le_target.classes_, ml_proba))
        
        debug_print(f"\n=== ML PREDICTION RESULTS ===")
        debug_print(f"Prediction probabilities:")
        for track, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
            debug_print(f"  {track}: {prob:.3f}")
        debug_print(f"ML prediction: {ml_track}")
        
    except Exception as e:
        debug_print(f"✗ ML prediction failed: {e}")
        ml_track = rule_prediction  # Fallback to rule-based
        debug_print(f"Using rule-based fallback: {ml_track}")

    # 10. Compare predictions and choose final result
    debug_print(f"\n=== FINAL COMPARISON ===")
    debug_print(f"Rule-based: {rule_prediction}")
    debug_print(f"ML model:   {ml_track}")
    if track_specialization:
        debug_print(f"Track specialization: {track_specialization}")
    
    # Enhanced decision logic: prefer rule-based for basic tracks (ML is biased toward BSCS)
    final_prediction = rule_prediction  # Use rule-based for basic tracks
    final_specialization = track_specialization
    
    # For basic tracks (BSIT, BSCS, BSCPE), trust rule-based logic since it's clearer
    debug_print(f"Using rule-based prediction for basic tracks: {rule_prediction}")
    if rule_prediction != ml_track:
        debug_print(f"ML disagreed ({ml_track}), but rule-based is more reliable for basic tracks")
    else:
        debug_print("Rule-based and ML predictions agree")
    
    debug_print(f"Final output: {final_prediction}")
    if final_specialization:
        debug_print(f"Final specialization: {final_specialization}")
    debug_print("=== ICT Track Prediction Complete ===")

    # 11. Output JSON with both prediction and scores for PHP
    # Use rule-based scores since we're using basic tracks
    result = {
        'recommended_track': final_prediction,
        'scores': rule_scores,
        'track_specialization': final_specialization
    }
    debug_print(f"Final JSON output: {result}")
    print(json.dumps(result))

except FileNotFoundError as e:
    debug_print(f"✗ File not found: {e}")
    # Return JSON fallback
    fallback_result = {
        'recommended_track': 'BSIT',
        'scores': {'BSCS': 0, 'BSIT': 1, 'BSCPE': 0},
        'track_specialization': 'Data Analytics'
    }
    print(json.dumps(fallback_result))
except json.JSONDecodeError as e:
    debug_print(f"✗ Invalid JSON: {e}")
    # Return JSON fallback
    fallback_result = {
        'recommended_track': 'BSIT',
        'scores': {'BSCS': 0, 'BSIT': 1, 'BSCPE': 0},
        'track_specialization': 'Data Analytics'
    }
    print(json.dumps(fallback_result))
except Exception as e:
    debug_print(f"✗ Unexpected error: {e}")
    debug_print(f"Error type: {type(e).__name__}")
    import traceback
    debug_print(f"Traceback: {traceback.format_exc()}")
    # Return JSON fallback
    fallback_result = {
        'recommended_track': 'BSIT',
        'scores': {'BSCS': 0, 'BSIT': 1, 'BSCPE': 0},
        'track_specialization': 'Data Analytics'
    }
    print(json.dumps(fallback_result))