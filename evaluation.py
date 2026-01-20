import json
import re
import os

# 設定類別權重 (可依需求調整)
ALPHA_COST = 2.0  # 詐騙樣本 (True) 的權重 (漏報代價大)
BETA_COST = 1.0   # 正常樣本 (False) 的權重

def calculate_dwa_from_jsonl(file_path):
    """
    從 JSONL 檔案讀取資料並計算衰減加權準確率 (DWA Score)
    
    參數:
    file_path (str): jsonl 檔案的路徑
    """
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return

    parsed_results = []
    global_max_len = 0
    valid_count = 0
    
    print(f"正在讀取檔案: {file_path} ...")
    
    # ---------------------------------------------------------
    # 步驟 1: 讀取檔案並預處理
    # ---------------------------------------------------------
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue 
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Line {line_idx+1} is not valid JSON. Skipped.")
                    continue

                # --- A. 提取對話長度 ---
                user_content = ""
                if 'messages' in item and isinstance(item['messages'], list):
                    for msg in item['messages']:
                        if msg.get('role') == 'user':
                            user_content = msg.get('content', "")
                            break
                
                match = re.search(r'<conversation>(.*?)</conversation>', user_content, re.DOTALL)
                if match:
                    actual_conversation = match.group(1).strip()
                else:
                    actual_conversation = user_content
                
                length = len(actual_conversation)
                if length > global_max_len:
                    global_max_len = length
                
                # --- B. 判斷答對與否 ---
                prediction = str(item.get('response', '')).strip()
                
                if 'labels' in item:
                    ground_truth = str(item['labels']).strip()
                elif 'label' in item:
                    ground_truth = str(item['label']) 
                else:
                    ground_truth = "Unknown"

                is_correct = (prediction.lower() == ground_truth.lower())
                
                parsed_results.append({
                    'line_no': line_idx + 1,
                    'length': length,
                    'is_correct': is_correct,
                    'prediction': prediction,
                    'ground_truth': ground_truth
                })
                valid_count += 1
                
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    if valid_count == 0:
        print("沒有讀取到有效資料。")
        return

    # ---------------------------------------------------------
    # 步驟 2: 計算 DWA 分數 (修改處)
    # ---------------------------------------------------------
    epsilon = 1e-9
    total_weighted_score = 0  
    total_possible_weight = 0 
    
    for i, res in enumerate(parsed_results):
        L = res['length']
        
        # 1. 計算長度衰減權重 (w_len)
        w_len = 1.0 - (L / (global_max_len + epsilon))
        w_len = max(0.0, w_len) 
        
        # 2. 計算類別成本權重 (w_class) [新增]
        # 判斷 Ground Truth 是否為詐騙 (True)
        is_fraud_sample = (res['ground_truth'].lower() == 'true' or res['ground_truth'] == '1')
        w_class = ALPHA_COST if is_fraud_sample else BETA_COST
        
        # 3. 結合權重 (Omega) [修改]
        final_weight = w_len * w_class
        
        # 4. 累加分數
        contribution = final_weight if res['is_correct'] else 0.0
        
        total_weighted_score += contribution
        total_possible_weight += final_weight

    # ---------------------------------------------------------
    # 步驟 3: 最終統計
    # ---------------------------------------------------------
    if total_possible_weight == 0:
        final_score = 0.0
    else:
        final_score = total_weighted_score / total_possible_weight

    print("=" * 80)
    print(f"📊 統計摘要 (DWA Metric):")
    print(f"   - 參數設定:          Alpha(Fraud)={ALPHA_COST}, Beta(Normal)={BETA_COST}")
    print(f"   - 總樣本數 (N):      {valid_count}")
    print(f"   - 全域最大長度 (Max): {global_max_len} chars")
    print(f"   - 加權總分 (Num):    {total_weighted_score:.4f}")
    print(f"   - 總權重 (Denom):    {total_possible_weight:.4f}")
    print("-" * 80)
    print(f"🎯 DWA Score (衰減加權準確率): {final_score:.4f}")
    print("=" * 80)
    
    return final_score


def qwen_8b_calculate_dwa_from_jsonl(file_path):
    """
    [Qwen版] 從 JSONL 檔案讀取資料並計算衰減加權準確率 (DWA Score)
    """
    
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return

    parsed_results = []
    global_max_len = 0
    valid_count = 0
    
    print(f"正在讀取檔案: {file_path} ...")
    
    # ---------------------------------------------------------
    # 步驟 1: 讀取檔案並預處理
    # ---------------------------------------------------------
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Line {line_idx+1} is not valid JSON. Skipped.")
                    continue

                # --- A. 提取對話長度 ---
                user_content = ""
                if 'messages' in item and isinstance(item['messages'], list):
                    for msg in item['messages']:
                        if msg.get('role') == 'user':
                            user_content = msg.get('content', "")
                            break
                
                match = re.search(r'<conversation>(.*?)</conversation>', user_content, re.DOTALL)
                if match:
                    actual_conversation = match.group(1).strip()
                else:
                    actual_conversation = user_content
                
                length = len(actual_conversation)
                if length > global_max_len:
                    global_max_len = length
                
                # --- B. 判斷答對與否 (Qwen 特殊處理) ---
                raw_response = str(item.get('response', '')).strip()
                prediction_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
                
                prediction_match = re.search(r'[01]', prediction_text)
                if prediction_match:
                    prediction = prediction_match.group(0)
                else:
                    prediction = prediction_text[:10] if prediction_text else "Unknown"
                
                if 'labels' in item:
                    ground_truth = str(item['labels']).strip()
                elif 'label' in item:
                    ground_truth = str(item['label']).strip()
                else:
                    ground_truth = "Unknown"

                is_correct = (prediction.lower() == ground_truth.lower())
                
                parsed_results.append({
                    'line_no': line_idx + 1,
                    'length': length,
                    'is_correct': is_correct,
                    'prediction': prediction,
                    'ground_truth': ground_truth
                })
                valid_count += 1
                
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    if valid_count == 0:
        print("沒有讀取到有效資料。")
        return

    # ---------------------------------------------------------
    # 步驟 2: 計算 DWA 分數 (修改處)
    # ---------------------------------------------------------
    epsilon = 1e-9
    total_weighted_score = 0  
    total_possible_weight = 0 
    
    for i, res in enumerate(parsed_results):
        L = res['length']
        
        # 1. 長度衰減
        w_len = 1.0 - (L / (global_max_len + epsilon))
        w_len = max(0.0, w_len)
        
        # 2. 類別成本 [新增]
        # 注意：Qwen 版本的 Ground Truth 可能是 "0"/"1" 或 "True"/"False"
        gt_str = res['ground_truth'].lower()
        is_fraud_sample = (gt_str == 'true' or gt_str == '1')
        w_class = ALPHA_COST if is_fraud_sample else BETA_COST
        
        # 3. 結合權重
        final_weight = w_len * w_class
        
        contribution = final_weight if res['is_correct'] else 0.0
        
        total_weighted_score += contribution
        total_possible_weight += final_weight

    # ---------------------------------------------------------
    # 步驟 3: 最終統計
    # ---------------------------------------------------------
    if total_possible_weight == 0:
        final_score = 0.0
    else:
        final_score = total_weighted_score / total_possible_weight

    print("=" * 80)
    print(f"📊 統計摘要 (DWA Metric - Qwen):")
    print(f"   - 參數設定:          Alpha(Fraud)={ALPHA_COST}, Beta(Normal)={BETA_COST}")
    print(f"   - 總樣本數 (N):      {valid_count}")
    print(f"   - 全域最大長度 (Max): {global_max_len} chars")
    print(f"   - 加權總分 (Num):    {total_weighted_score:.4f}")
    print(f"   - 總權重 (Denom):    {total_possible_weight:.4f}")
    print("-" * 80)
    print(f"🎯 DWA Score (衰減加權準確率): {final_score:.4f}")
    print("=" * 80)
    
    return final_score


def oss_calculate_dwa_from_jsonl(file_path):
    """
    [OSS版] 從 JSONL 檔案讀取資料並計算衰減加權準確率 (DWA Score)
    """
    
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return

    parsed_results = []
    global_max_len = 0
    valid_count = 0
    no_match_count = 0
    
    print(f"正在讀取檔案: {file_path} ...")
    
    # ---------------------------------------------------------
    # 步驟 1: 讀取檔案並預處理
    # ---------------------------------------------------------
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Line {line_idx+1} is not valid JSON. Skipped.")
                    continue

                # --- A. 提取對話長度 ---
                user_content = ""
                if 'messages' in item and isinstance(item['messages'], list):
                    for msg in item['messages']:
                        if msg.get('role') == 'user':
                            user_content = msg.get('content', "")
                            break
                
                match = re.search(r'<conversation>(.*?)</conversation>', user_content, re.DOTALL)
                if match:
                    actual_conversation = match.group(1).strip()
                else:
                    actual_conversation = user_content
                
                length = len(actual_conversation)
                if length > global_max_len:
                    global_max_len = length
                
                # --- B. 判斷答對與否 (OSS 特殊處理) ---
                raw_response = str(item.get('response', '')).strip()
                true_false_matches = list(re.finditer(r'\b(True|False)\b', raw_response, re.IGNORECASE))
                
                if true_false_matches:
                    last_match = true_false_matches[-1]
                    prediction = last_match.group(1)
                else:
                    prediction = "Unknown"
                    no_match_count += 1
                
                if 'labels' in item:
                    ground_truth = str(item['labels']).strip()
                elif 'label' in item:
                    label_val = item['label']
                    if label_val == 0:
                        ground_truth = "False"
                    elif label_val == 1:
                        ground_truth = "True"
                    else:
                        ground_truth = str(label_val)
                else:
                    ground_truth = "Unknown"

                is_correct = (prediction.lower() == ground_truth.lower())
                
                parsed_results.append({
                    'line_no': line_idx + 1,
                    'length': length,
                    'is_correct': is_correct,
                    'prediction': prediction,
                    'ground_truth': ground_truth
                })
                valid_count += 1
                
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    if valid_count == 0:
        print("沒有讀取到有效資料。")
        return

    if no_match_count > 0:
        print(f"⚠️  警告: 有 {no_match_count} 筆資料的 response 中未找到 True 或 False")

    # ---------------------------------------------------------
    # 步驟 2: 計算 DWA 分數 (修改處)
    # ---------------------------------------------------------
    epsilon = 1e-9
    total_weighted_score = 0  
    total_possible_weight = 0 
    
    for i, res in enumerate(parsed_results):
        L = res['length']
        
        # 1. 長度衰減
        w_len = 1.0 - (L / (global_max_len + epsilon))
        w_len = max(0.0, w_len)
        
        # 2. 類別成本 [新增]
        # 判斷 Ground Truth 是否為詐騙 (True)
        gt_str = res['ground_truth'].lower()
        is_fraud_sample = (gt_str == 'true' or gt_str == '1')
        w_class = ALPHA_COST if is_fraud_sample else BETA_COST
        
        # 3. 結合權重
        final_weight = w_len * w_class
        
        contribution = final_weight if res['is_correct'] else 0.0
        
        total_weighted_score += contribution
        total_possible_weight += final_weight

    # ---------------------------------------------------------
    # 步驟 3: 最終統計
    # ---------------------------------------------------------
    if total_possible_weight == 0:
        final_score = 0.0
    else:
        final_score = total_weighted_score / total_possible_weight

    correct_count = sum(1 for res in parsed_results if res['is_correct'])
    accuracy = correct_count / valid_count if valid_count > 0 else 0.0

    print("=" * 80)
    print(f"📊 統計摘要 (DWA Metric - OSS):")
    print(f"   - 參數設定:          Alpha(Fraud)={ALPHA_COST}, Beta(Normal)={BETA_COST}")
    print(f"   - 總樣本數 (N):      {valid_count}")
    print(f"   - 答對數量:          {correct_count}")
    print(f"   - 傳統準確率:        {accuracy:.4f}")
    print(f"   - 全域最大長度 (Max): {global_max_len} chars")
    print(f"   - 加權總分 (Num):    {total_weighted_score:.4f}")
    print(f"   - 總權重 (Denom):    {total_possible_weight:.4f}")
    print("-" * 80)
    print(f"🎯 DWA Score (衰減加權準確率): {final_score:.4f}")
    print("=" * 80)
    
    return final_score

"""
print("base_8b")
calculate_cdi_from_jsonl("./inference_data/base_8b_infer_all_test_results.jsonl")
print("sft_8b")
calculate_cdi_from_jsonl("./inference_data/sft_8b_infer_all_test_results_50_v3.jsonl")
print("base_70b_awq")
calculate_cdi_from_jsonl("./inference_data/base_70b_awq_infer_all_test_results.jsonl")
print("qwen_8b")
qwen_8b_calculate_cdi_from_jsonl("./inference_data/qwen_8b_infer_all_test_results.jsonl")
print("ministral_8b")
calculate_cdi_from_jsonl("./inference_data/ministral_8b_infer_all_test_results.jsonl")
print("ministral_8b_v1_50")
calculate_cdi_from_jsonl("./inference_data/ministral_8b_infer_all_test_results_50_v1.jsonl")
print("ministral_8b_v1_81")
calculate_cdi_from_jsonl("./inference_data/ministral_8b_infer_all_test_results_81_v1.jsonl")
print("ministral_8b_v2_30")
calculate_cdi_from_jsonl("./inference_data/ministral_8b_infer_all_test_results_30_v2.jsonl")
"""
print("ministral_8b_v1_50")
calculate_dwa_from_jsonl("./inference_data/ministral_8b_infer_test_results_50_v1.jsonl")
print("ministral_8b_v1_81")
calculate_dwa_from_jsonl("./inference_data/ministral_8b_infer_test_results_81_v1.jsonl")
print("ministral_8b")
calculate_dwa_from_jsonl("./inference_data/ministral_8b_infer_test_results.jsonl")
print("qwen_8b")
qwen_8b_calculate_dwa_from_jsonl("./inference_data/qwen_8b_infer_test_results.jsonl")
print("qwen_32b")
qwen_8b_calculate_dwa_from_jsonl("./inference_data/qwen_32b_infer_test_results.jsonl")
print("base_8b")
calculate_dwa_from_jsonl("./inference_data/base_8b_infer_test_result.jsonl")
print("sft_8b")
calculate_dwa_from_jsonl("./inference_data/sft_8b_infer_test_results_50_v3.jsonl")
print("base_70b_awq")
calculate_dwa_from_jsonl("./inference_data/base_70b_awq_infer_test_results.jsonl")
print("gpt_120b")
oss_calculate_dwa_from_jsonl("./inference_data/gpt_120b_infer_test_results.jsonl")
print("sft_8b_v4")
calculate_dwa_from_jsonl("./inference_data/sft_8b_infer_test_results_20_v4.jsonl")
calculate_dwa_from_jsonl("./inference_data/sft_8b_infer_test_results_40_v4.jsonl")
calculate_dwa_from_jsonl("./inference_data/sft_8b_infer_test_results_60_v4.jsonl")
calculate_dwa_from_jsonl("./inference_data/sft_8b_infer_test_results_80_v4.jsonl")
calculate_dwa_from_jsonl("./inference_data/sft_8b_infer_test_results_100_v4.jsonl")
calculate_dwa_from_jsonl("./inference_data/sft_8b_infer_test_results_108_v4.jsonl")