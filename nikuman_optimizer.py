"""中華まん製造数の自動提案 Ver.2（仮想需要モデル）
実行: python nikuman_optimizer.py
下の「入力設定」を変更して再実行してください。外部ライブラリ不要。
実データに基づく学習済みモデルではありません。
"""
import math
from dataclasses import dataclass

# ================ 入力設定：ここを変更 ================
TEMPERATURE_C = 27        # 最高気温（℃）
DAY_OF_WEEK = '日'         # 月,火,水,木,金,土,日
RAIN = False              # 雨なら True
CAPACITY = 18             # ショーケース容量
INITIAL_STOCK = 0         # 7:00時点の完成済み在庫（期限情報不明なら0推奨）
INITIAL_REMAINING_MIN = 180  # 初期在庫の残り販売可能時間
WASTE_COST = 160          # 廃棄1個の損失（仮の円）
STOCKOUT_COST = 200       # 欠品1個の機会損失（仮の円）
# 7:00〜翌2:00、1時間ごとの基本需要（仮定・個/時間）。自由に変更可能。
BASE_HOURLY = [1.0, 1.4, 1.6, 2.0, 2.6, 3.0, 2.0, 1.6, 1.8, 2.4,
               2.5, 2.0, 1.4, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
# =======================================================

START = 7 * 60
LAST_PRODUCTION = 18 * 60
END = 26 * 60
LEAD = 50
SHELF = 430
STEP = 10
DECISION_INTERVAL = 30


def clock(minute):
    return f'{minute // 60 % 24:02d}:{minute % 60:02d}' + (' 翌日' if minute >= 1440 else '')


def demand_multiplier():
    # すべて例示用の仮係数。実測データがあれば推定値に置き換える。
    temp = max(0.40, min(1.50, 1.0 - 0.035 * (TEMPERATURE_C - 18)))
    weekday = 0.82 if DAY_OF_WEEK == '日' else (1.08 if DAY_OF_WEEK == '土' else 1.0)
    rain = 0.85 if RAIN else 1.0
    return temp * weekday * rain


def rate(t):
    i = (t - START) // 60
    return BASE_HOURLY[i] * demand_multiplier() if 0 <= i < len(BASE_HOURLY) else 0.0


def forecast(a, b):
    return sum(rate(t) * STEP / 60 for t in range(a, min(b, END), STEP))


def poisson_probs(lam):
    """P(X=k) を尾部確率が十分小さくなるまで列挙。"""
    p = math.exp(-lam)
    probs = [p]
    for k in range(1, max(20, math.ceil(lam + 12 * math.sqrt(lam + 1)))):
        p *= lam / k
        probs.append(p)
    probs[-1] += max(0.0, 1 - sum(probs))
    return probs


def expected_shortage(lam, stock):
    return sum(max(0, k - stock) * p for k, p in enumerate(poisson_probs(lam)))


def expected_leftover(lam, stock):
    return sum(max(0, stock - k) * p for k, p in enumerate(poisson_probs(lam)))


@dataclass
class Batch:
    ready: int
    qty: int
    expiry: int


def choose_quantity(now, inventory, pending):
    ready = now + LEAD
    # 完成時点の見込み在庫を、期限切れ・販売・仕込み中を考慮して概算
    valid = sum(1 for exp in inventory if exp > ready)
    valid += sum(b.qty for b in pending if b.ready <= ready and b.expiry > ready)
    expected_sales_before = forecast(now, ready)
    projected = max(0, round(valid - expected_sales_before))
    max_qty = max(0, CAPACITY - projected)
    # 2時間先の需要と、新たな製造分の期限までの需要を評価
    near_end = min(ready + 120, END)
    near_demand = forecast(ready, near_end)
    lifetime_demand = forecast(ready, min(ready + SHELF, END))
    best = (float('inf'), 0)
    for q in range(max_qty + 1):
        total = projected + q
        # 近い2時間の欠品リスク + 新規製造分が売れ残るリスク
        shortage = expected_shortage(near_demand, total)
        # 既存在庫が先に売れるため、新規製造分の期待残数を概算
        waste = expected_leftover(lifetime_demand, total) - expected_leftover(lifetime_demand, projected)
        loss = STOCKOUT_COST * shortage + WASTE_COST * max(0, waste)
        # 同点では少ない製造数を優先
        if loss < best[0] - 1e-8:
            best = (loss, q)
    return best[1]


def run():
    if len(BASE_HOURLY) != (END - START) // 60:
        raise ValueError('BASE_HOURLY は19個の値が必要です')
    if DAY_OF_WEEK not in '月火水木金土日':
        raise ValueError('DAY_OF_WEEK は月〜日の漢字1文字にしてください')
    if not (0 <= INITIAL_STOCK <= CAPACITY) or CAPACITY < 1:
        raise ValueError('在庫・容量の設定を確認してください')
    if WASTE_COST < 0 or STOCKOUT_COST < 0:
        raise ValueError('損失は0以上にしてください')

    inventory = [START + INITIAL_REMAINING_MIN] * INITIAL_STOCK
    pending = []
    plan = []
    expected_sold = 0.0
    expected_missed = 0.0
    expected_waste = 0.0
    for now in range(START, END, STEP):
        # 期限切れ
        expired = [e for e in inventory if e <= now]
        expected_waste += len(expired)
        inventory = [e for e in inventory if e > now]
        # 製造完了
        arriving = [b for b in pending if b.ready == now]
        pending = [b for b in pending if b.ready != now]
        for b in arriving:
            free = max(0, CAPACITY - len(inventory))
            inventory.extend([b.expiry] * min(free, b.qty))
            expected_waste += max(0, b.qty - free)  # 容量超過分も損失として計上
        inventory.sort()

        if now <= LAST_PRODUCTION and (now - START) % DECISION_INTERVAL == 0:
            q = choose_quantity(now, inventory, pending)
            if q:
                pending.append(Batch(now + LEAD, q, now + LEAD + SHELF))
                plan.append((now, now + LEAD, q))

        # 平均需要で在庫を消費する決定論的な簡易検証（端数を持ち越す）
        # 需要を確率的に扱うのは製造判断側。こちらは参考用の平均需要経路。
        expected_missed += 0  # 以下の在庫消費は run 内の累積需要で実施
        # 10分間の期待需要を小数として累積し、1個に達したら購入発生
        if now == START:
            carry = 0.0
        carry += rate(now) * STEP / 60
        requests = math.floor(carry + 1e-10)
        carry -= requests
        sold = min(requests, len(inventory))
        del inventory[:sold]
        expected_sold += sold
        expected_missed += requests - sold

    expected_waste += len(inventory)
    print('\n=== 中華まん製造提案 Ver.2（仮定データ） ===')
    print(f'条件: {DAY_OF_WEEK}曜日 / {TEMPERATURE_C}℃ / 雨={RAIN} / 容量={CAPACITY}個')
    print(f'予測需要合計: {forecast(START, END):.1f}個')
    print(f'損失設定: 廃棄{WASTE_COST}円/個、欠品{STOCKOUT_COST}円/個')
    print('\n【自動提案された製造計画】')
    if not plan:
        print('追加製造なし')
    for t, ready, q in plan:
        print(f'{clock(t)} 製造開始 → {clock(ready)} 完成 : {q}個')
    print(f'\n製造合計: {sum(q for _, _, q in plan)}個')
    print(f'参考・平均需要経路での販売: {expected_sold:.0f}個 / 欠品: {expected_missed:.0f}個 / 廃棄: {expected_waste:.0f}個')
    print('※ 販売・欠品・廃棄は仮想需要1経路の参考値。実際の平均や保証値ではありません。')
    print('※ 製造数は近似的な局所最適化。全日の厳密な最適解ではありません。')
    print('※ 完成後430分の期限、50分の製造時間、18時までの製造開始を反映。')


if __name__ == '__main__':
    run()
