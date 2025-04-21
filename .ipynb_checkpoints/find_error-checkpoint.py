{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4040db76-9a61-48fb-89f8-f7d2d1b78be3",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'ace_tools'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[16], line 4\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpandas\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[0;32m----> 4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mace_tools\u001b[39;00m\n\u001b[1;32m      6\u001b[0m \u001b[38;5;66;03m# PKL 경로\u001b[39;00m\n\u001b[1;32m      7\u001b[0m PKL_PATH \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdata/road_TSP_100.pkl\u001b[39m\u001b[38;5;124m\"\u001b[39m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'ace_tools'"
     ]
    }
   ],
   "source": [
    "# find_error.py\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "PKL_PATH = \"data/road_TSP_100.pkl\"  # 실제 경로에 맞게 수정하세요\n",
    "\n",
    "def main():\n",
    "    # 1) 데이터 로드\n",
    "    with open(PKL_PATH, \"rb\") as f:\n",
    "        data = pickle.load(f)\n",
    "\n",
    "    total_fallback = 0\n",
    "    n_instances   = len(data)\n",
    "\n",
    "    # 2) 인스턴스별 fallback 개수 집계\n",
    "    for idx, inst in enumerate(data):\n",
    "        dist = np.array(inst[\"dist\"], dtype=float)\n",
    "        # 60 000m 로 채워진 엔트리 개수\n",
    "        cnt = int((dist == 60000.0).sum())\n",
    "        total_fallback += cnt\n",
    "        print(f\"Instance {idx:3d}: fallback entries = {cnt}\")\n",
    "\n",
    "    # 3) 요약 정보 출력\n",
    "    avg = total_fallback / n_instances if n_instances else 0\n",
    "    print(\"\\n\" + \"-\"*40)\n",
    "    print(f\"Total fallback entries : {total_fallback}\")\n",
    "    print(f\"Instances              : {n_instances}\")\n",
    "    print(f\"Average per instance   : {avg:.2f}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
