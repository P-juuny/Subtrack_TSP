import requests
import pandas as pd
import numpy as np
import os
import time
import json
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm

def get_its_traffic_data(api_key, start_date, end_date, regions=None, output_dir="traffic_data", 
                         time_slots=None, link_id=None, historical=True):
    """
    국가교통정보센터(ITS) API를 사용하여 교통 정보를 수집
    
    Args:
        api_key (str): ITS API 키
        start_date (str): 시작 날짜 (YYYYMMDD 형식)
        end_date (str): 종료 날짜 (YYYYMMDD 형식)
        regions (list): 특정 지역 코드 목록 (None이면 기본 지역)
        output_dir (str): 데이터 저장 디렉토리
        time_slots (list): 수집할 특정 시간대 목록 (None이면 모든 시간대)
        link_id (list): 특정 링크 ID 목록 (None이면 모든 링크)
        historical (bool): 과거 데이터 수집 여부 (True면 과거, False면 실시간)
    """
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 날짜 범위 생성
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    date_range = [(start + timedelta(days=i)).strftime("%Y%m%d") 
                  for i in range((end - start).days + 1)]
    
    # 시간대 범위
    if time_slots:
        hours = [f"{h:02d}" for h in time_slots]
    else:
        hours = [f"{h:02d}" for h in range(24)]
    
    # 기본 지역 설정 (서울시 주요 도로)
    if not regions:
        regions = ["1160", "1170", "1180"]  # 서울특별시 코드
    
    # 데이터 수집
    all_data = []
    
    # API 엔드포인트 설정
    if historical:
        base_url = "http://openapi.its.go.kr:8082/api/NTrafficHistory"
    else:
        base_url = "http://openapi.its.go.kr:8082/api/NTrafficInfo"
    
    # 링크 정보 수집 (도로 ID와 이름)
    link_info = get_link_info(api_key)
    if link_info is None:
        print("링크 정보를 가져오는데 실패했습니다. 진행합니다...")
    else:
        print(f"총 {len(link_info)} 개의 링크 정보를 가져왔습니다.")
    
    # 지역별로 데이터 수집
    for region in regions:
        print(f"\n지역 {region}의 데이터 수집 중...")
        
        for date in tqdm(date_range, desc="날짜 처리"):
            for hour in hours:
                print(f"  {date} {hour}:00 데이터 수집 중...")
                
                # API 매개변수 설정
                params = {
                    "key": api_key,
                    "type": "json",
                    "ReqDate": date,
                    "RegionCode": region
                }
                
                if historical:
                    params["HourDev"] = hour
                
                if link_id:
                    params["LinkID"] = ','.join(link_id)
                
                try:
                    response = requests.get(base_url, params=params)
                    
                    if response.status_code != 200:
                        print(f"  오류 응답: {response.status_code}, {response.text[:200]}...")
                        continue
                    
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        print(f"  JSON 디코딩 오류. 응답: {response.text[:200]}...")
                        continue
                    
                    # API 응답 구조 확인
                    if 'body' in data and 'items' in data['body']:
                        traffic_items = data['body']['items']
                        
                        if not traffic_items:
                            print(f"  {date} {hour}:00에 데이터가 없습니다.")
                            continue
                        
                        print(f"  {len(traffic_items)}개의 교통 데이터를 찾았습니다.")
                        
                        # 각 항목 처리
                        for item in traffic_items:
                            # 링크 정보 매핑 (있는 경우)
                            link_name = "Unknown"
                            if link_info and 'linkId' in item:
                                link_id = item['linkId']
                                if link_id in link_info:
                                    link_name = link_info[link_id]
                            
                            # 필요한 필드 추출
                            traffic_data = {
                                'link_id': item.get('linkId', ''),
                                'link_name': link_name,
                                'speed': item.get('speed', 0),  # 평균 속도
                                'travel_time': item.get('traveltime', 0),  # 통행 시간
                                'congestion': item.get('lvl', 0),  # 혼잡도 (1:원활, 2:서행, 3:정체)
                                'region_code': region,
                                'date': date,
                                'hour': hour
                            }
                            
                            all_data.append(traffic_data)
                    else:
                        print(f"  예상치 못한 데이터 구조: {list(data.keys())}")
                    
                    # API 요청 제한 방지
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  {date} {hour}:00 데이터 수집 중 오류 발생: {e}")
                    time.sleep(1)  # 오류 발생 시 더 오래 대기
    
    # 데이터프레임 변환 및 저장
    if all_data:
        df = pd.DataFrame(all_data)
        
        # 수치형으로 변환
        for col in ['speed', 'travel_time', 'congestion']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 날짜 및 시간 정보 추가
        df['datetime'] = pd.to_datetime(df['date'] + df['hour'], format='%Y%m%d%H')
        
        # 출력 파일명 생성
        region_str = '_'.join(regions)
        data_type = "historical" if historical else "realtime"
        output_file = os.path.join(output_dir, f"its_{data_type}_traffic_{region_str}_{start_date}_to_{end_date}.csv")
        
        # CSV로 저장
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n데이터 수집 완료. {output_file}에 저장되었습니다.")
        
        # 링크 정보 매핑 파일 저장
        link_df = df[['link_id', 'link_name']].drop_duplicates()
        link_df.to_csv(os.path.join(output_dir, "link_mapping.csv"), index=False, encoding='utf-8-sig')
        
        # PKL 형식으로도 저장
        pkl_output_file = output_file.replace('.csv', '.pkl')
        df.to_pickle(pkl_output_file)
        print(f"데이터가 PKL 형식으로도 저장되었습니다: {pkl_output_file}")
        
        return df
    else:
        print("수집된 데이터가 없습니다.")
        return None

def get_link_info(api_key):
    """
    국가교통정보센터(ITS) API에서 링크 정보를 가져옴
    
    Args:
        api_key (str): ITS API 키
    
    Returns:
        dict: 링크 ID를 키로, 링크 이름을 값으로 하는 딕셔너리
    """
    # 링크 정보 API URL
    url = "http://openapi.its.go.kr:8082/api/NLinkInfo"
    
    params = {
        "key": api_key,
        "type": "json"
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"링크 정보 가져오기 실패: {response.status_code}")
            return None
        
        data = response.json()
        
        if 'body' in data and 'items' in data['body']:
            link_items = data['body']['items']
            
            # 링크 ID와 이름을 매핑
            link_info = {}
            for item in link_items:
                link_id = item.get('linkId', '')
                link_name = item.get('roadName', 'Unknown')
                if link_id:
                    link_info[link_id] = link_name
            
            return link_info
        else:
            print("링크 정보 데이터 구조가 예상과 다릅니다.")
            return None
    
    except Exception as e:
        print(f"링크 정보 가져오기 중 오류 발생: {e}")
        return None

def create_traffic_matrix(traffic_df, output_dir="traffic_matrices", region=None):
    """
    수집된 교통 데이터를 기반으로 시간대별 교통량 행렬 생성
    
    Args:
        traffic_df (DataFrame): 수집된 교통 데이터
        output_dir (str): 출력 디렉토리
        region (str): 특정 지역 코드 (None이면 모든 데이터 처리)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 특정 지역만 필터링
    if region and 'region_code' in traffic_df.columns:
        traffic_df = traffic_df[traffic_df['region_code'] == region]
    
    # 유니크한 날짜와 시간대 추출
    if 'date' in traffic_df.columns and 'hour' in traffic_df.columns:
        dates = traffic_df['date'].unique()
        hours = traffic_df['hour'].unique()
    else:
        print("Warning: 'date' 또는 'hour' 열이 데이터에 없습니다.")
        return None
    
    # 링크 ID 매핑
    if 'link_id' in traffic_df.columns:
        link_ids = traffic_df['link_id'].unique().tolist()
        n_links = len(link_ids)
        link_id_to_idx = {link_id: i for i, link_id in enumerate(link_ids)}
    else:
        print("Warning: 'link_id' 열이 데이터에 없습니다.")
        return None
    
    print(f"{len(dates)}개 날짜, {len(hours)}개 시간대에 대한 교통 행렬 생성 중...")
    print(f"총 링크 수: {n_links}")
    
    # 링크 매핑 정보 저장
    link_mapping = pd.DataFrame({
        'index': range(n_links),
        'link_id': link_ids,
        'link_name': [traffic_df[traffic_df['link_id'] == link_id]['link_name'].iloc[0] 
                       if not traffic_df[traffic_df['link_id'] == link_id].empty else "Unknown" 
                       for link_id in link_ids]
    })
    
    # 저장 경로에 지역 정보 추가
    if region:
        output_dir = os.path.join(output_dir, region)
        os.makedirs(output_dir, exist_ok=True)
        link_mapping.to_csv(os.path.join(output_dir, f"link_mapping_{region}.csv"), index=False)
    else:
        link_mapping.to_csv(os.path.join(output_dir, "link_mapping.csv"), index=False)
    
    # 날짜 및 시간대별 행렬 생성
    matrices_created = 0
    
    for date in dates:
        for hour in hours:
            # 해당 날짜 및 시간대의 데이터 필터링
            day_data = traffic_df[(traffic_df['date'] == date) & (traffic_df['hour'] == hour)]
            
            if day_data.empty:
                continue
            
            # 행렬 초기화 (링크 간 거리 조정을 위한 계수)
            traffic_matrix = np.ones((n_links, n_links))
            
            # 교통량 데이터 채우기
            for _, row in day_data.iterrows():
                link_id = row['link_id']
                if link_id in link_id_to_idx:
                    idx = link_id_to_idx[link_id]
                    
                    # 교통 상황에 따른 계수 설정
                    if pd.notna(row['speed']) and row['speed'] > 0:
                        speed = float(row['speed'])
                        # 속도를 계수로 변환 (속도가 빠를수록 계수가 낮음)
                        max_speed = 100.0  # 기준 최대 속도
                        factor = max(0.8, min(2.0, max_speed / max(speed, 1.0)))
                    elif pd.notna(row['congestion']) and row['congestion'] > 0:
                        congestion = int(row['congestion'])
                        # 혼잡도를 계수로 변환 (1:원활=0.8, 2:서행=1.2, 3:정체=1.6)
                        if congestion == 1:  # 원활
                            factor = 0.8
                        elif congestion == 2:  # 서행
                            factor = 1.2
                        elif congestion == 3:  # 정체
                            factor = 1.6
                        else:
                            factor = 1.0
                    else:
                        factor = 1.0
                    
                    # 해당 링크의 모든 연결에 대해 가중치 적용
                    traffic_matrix[idx, :] *= factor
                    traffic_matrix[:, idx] *= factor
            
            # NumPy 배열로 저장
            filename = f"traffic_matrix_{date}_{hour}.npy"
            output_file = os.path.join(output_dir, filename)
            np.save(output_file, traffic_matrix)
            matrices_created += 1
    
    print(f"모든 교통 행렬이 {output_dir}에 저장되었습니다 (총 {matrices_created}개)")
    return link_mapping

def convert_to_tsp_dataset(traffic_df, link_mapping, base_pkl_path, output_pkl, region=None):
    """
    교통 데이터를 TSP 학습용 데이터셋으로 변환
    
    Args:
        traffic_df (DataFrame): 수집된 교통 데이터
        link_mapping (DataFrame): 링크 매핑 정보
        base_pkl_path (str): 기본 TSP 데이터셋 PKL 파일 경로
        output_pkl (str): 출력 PKL 파일 경로
        region (str): 특정 지역 코드 (None이면 모든 데이터 처리)
    """
    import pickle
    import torch
    
    # 기본 데이터 로드
    try:
        with open(base_pkl_path, 'rb') as f:
            base_data = pickle.load(f)
        print(f"{base_pkl_path}에서 기본 TSP 데이터를 로드했습니다.")
    except Exception as e:
        print(f"기본 PKL 파일 로드 중 오류 발생: {e}")
        return
    
    # 특정 지역만 필터링
    if region and 'region_code' in traffic_df.columns:
        traffic_df = traffic_df[traffic_df['region_code'] == region]
    
    # 날짜/시간별로 그룹화
    grouped = traffic_df.groupby(['date', 'hour'])
    
    # 새 데이터셋 생성
    tsp_dataset = []
    
    for (date, hour), group in tqdm(grouped, desc="TSP 데이터셋 생성"):
        print(f"{date} {hour}:00 처리 중...")
        
        # 각 그룹에 대해 TSP 인스턴스 생성
        for i, base_instance in enumerate(base_data[:min(10, len(base_data))]):  # 예시로 10개만
            new_instance = {}
            
            # 기본 인스턴스 복사
            for key, val in base_instance.items():
                if isinstance(val, torch.Tensor):
                    new_instance[key] = val.clone()
                else:
                    new_instance[key] = val
            
            # 교통량 정보 추가
            traffic_factors = {}
            for _, row in group.iterrows():
                traffic_factors[row['link_id']] = {
                    'speed': row['speed'],
                    'congestion': row['congestion']
                }
            
            new_instance['traffic'] = traffic_factors
            new_instance['date'] = date
            new_instance['hour'] = hour
            
            # 원본 거리 행렬 복사
            if 'dist' in new_instance:
                new_instance['orig_dist'] = new_instance['dist'].clone()
                
                # 교통량에 따른 거리 조정
                for link_id, factors in traffic_factors.items():
                    if 'link_id' in link_mapping.columns and link_id in link_mapping['link_id'].values:
                        idx = link_mapping[link_mapping['link_id'] == link_id]['index'].iloc[0]
                        
                        # 교통 상황에 따른 계수 계산
                        if pd.notna(factors['speed']) and factors['speed'] > 0:
                            speed = float(factors['speed'])
                            # 속도를 계수로 변환 (속도가 빠를수록 계수가 낮음)
                            max_speed = 100.0  # 기준 최대 속도
                            factor = max(0.8, min(2.0, max_speed / max(speed, 1.0)))
                        elif pd.notna(factors['congestion']) and factors['congestion'] > 0:
                            congestion = int(factors['congestion'])
                            # 혼잡도를 계수로 변환 (1:원활=0.8, 2:서행=1.2, 3:정체=1.6)
                            if congestion == 1:  # 원활
                                factor = 0.8
                            elif congestion == 2:  # 서행
                                factor = 1.2
                            elif congestion == 3:  # 정체
                                factor = 1.6
                            else:
                                factor = 1.0
                        else:
                            factor = 1.0
                        
                        # 거리 행렬 조정
                        if idx < len(new_instance['dist']):
                            new_instance['dist'][idx, :] *= factor
                            new_instance['dist'][:, idx] *= factor
            
            tsp_dataset.append(new_instance)
    
    # PKL로 저장
    with open(output_pkl, 'wb') as f:
        pickle.dump(tsp_dataset, f)
    
    print(f"교통 데이터가 포함된 TSP 데이터셋이 생성되었습니다. {output_pkl}에 저장되었습니다.")
    print(f"총 인스턴스 수: {len(tsp_dataset)}")

def main():
    parser = argparse.ArgumentParser(description='국가교통정보센터(ITS) API 데이터를 수집하고 TSP 학습용으로 변환')
    parser.add_argument('--api_key', type=str, required=True, help='ITS API 키')
    parser.add_argument('--start_date', type=str, required=True, help='시작 날짜 (YYYYMMDD 형식)')
    parser.add_argument('--end_date', type=str, required=True, help='종료 날짜 (YYYYMMDD 형식)')
    parser.add_argument('--output_dir', type=str, default='traffic_data', help='데이터 저장 디렉토리')
    parser.add_argument('--regions', type=str, nargs='+', help='수집할 특정 지역 코드 목록')
    parser.add_argument('--time_slots', type=int, nargs='+', 
                        help='수집할 특정 시간대 (0-23)')
    parser.add_argument('--link_id', type=str, nargs='+', help='수집할 특정 링크 ID 목록')
    parser.add_argument('--realtime', action='store_true', 
                        help='실시간 데이터 수집 (기본값: 과거 데이터)')
    parser.add_argument('--create_matrices', action='store_true', 
                        help='교통량 행렬 생성')
    parser.add_argument('--base_pkl', type=str, help='기본 TSP 데이터셋 PKL 파일 경로')
    parser.add_argument('--convert_to_tsp', action='store_true', 
                        help='교통 데이터를 TSP 학습용으로 변환')
    
    args = parser.parse_args()
    
    # 데이터 수집
    traffic_df = get_its_traffic_data(
        api_key=args.api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        regions=args.regions,
        output_dir=args.output_dir,
        time_slots=args.time_slots,
        link_id=args.link_id,
        historical=not args.realtime
    )
    
    if traffic_df is not None:
        # 교통량 행렬 생성
        if args.create_matrices:
            if args.regions:
                link_mappings = {}
                for region in args.regions:
                    link_mapping = create_traffic_matrix(
                        traffic_df,
                        os.path.join(args.output_dir, "matrices"),
                        region=region
                    )
                    if link_mapping is not None:
                        link_mappings[region] = link_mapping
                    
                    # TSP 데이터셋으로 변환
                    if args.convert_to_tsp and args.base_pkl and link_mapping is not None:
                        output_pkl = os.path.join(args.output_dir, f"road_TSP_traffic_{region}.pkl")
                        convert_to_tsp_dataset(
                            traffic_df,
                            link_mapping,
                            args.base_pkl,
                            output_pkl,
                            region=region
                        )
            else:
                link_mapping = create_traffic_matrix(
                    traffic_df,
                    os.path.join(args.output_dir, "matrices")
                )
                
                # TSP 데이터셋으로 변환
                if args.convert_to_tsp and args.base_pkl and link_mapping is not None:
                    output_pkl = os.path.join(args.output_dir, "road_TSP_traffic.pkl")
                    convert_to_tsp_dataset(
                        traffic_df,
                        link_mapping,
                        args.base_pkl,
                        output_pkl
                    )

if __name__ == "__main__":
    main()