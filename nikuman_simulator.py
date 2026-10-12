"""中華まんの製造・廃棄シミュレーター（仮想データ）
実行: python nikuman_simulator.py
外部ライブラリ不要。実店舗の販売実績ではありません。
"""
import math
import random
from dataclasses import dataclass

START = 7 * 60
LAST_START = 18 * 60
END = 26 * 60  # 翌2時まで追跡
LEAD = 50
SHELF = 430
CAPACITY = 18
STEP = 10
TRIALS = 300
# 7時から翌2時まで、各時刻の「1時間あたり期待販売数」（仮定）
HOURLY = [1.0, 1.4, 1.6, 2.0, 2.6, 3.0, 2.0, 1.6, 1.8, 2.4,
          2.5, 2.0, 1.4, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2]


def expected_demand(start, end):
    """区間 [start,end) の予測需要。"""
    total = 0.0
    for t in range(start, end, STEP):
        h = (t - START) // 60
        if 0 <= h < len(HOURLY):
            total += HOURLY[h] * STEP / 60
    return total


def poisson(lam, rng):
    # 小さいlambda用の厳密なポアソン乱数
    p, k, limit = 1.0, 0, math.exp(-lam)
    while p > limit:
        p *= rng.random()
        k += 1
    return k - 1


@dataclass
class Result:
    produced: int = 0
    sold: int = 0
    wasted: int = 0
    missed: int = 0
    rejected: int = 0


def run_day(policy, seed):
    rng = random.Random(seed)
    inventory = []  # 各個体の廃棄時刻
    pending = []    # (完成時刻, 個数)
    result = Result()
    fixed_plan = {START: 8, START + 180: 8, START + 360: 8, START + 540: 8}

    for now in range(START, END, STEP):
        # 完成した商品の補充。容量を超える製造は受け入れられないと仮定
        arrivals = [(t, q) for t, q in pending if t == now]
        pending = [(t, q) for t, q in pending if t != now]
        for _, qty in arrivals:
            free = CAPACITY - len(inventory)
            accepted = min(qty, free)
            inventory.extend([now + SHELF] * accepted)
            result.produced += qty
            result.rejected += qty - accepted
        # 期限切れの除去
        result.wasted += sum(exp <= now for exp in inventory)
        inventory = sorted(exp for exp in inventory if exp > now)

        if now <= LAST_START:
            if policy == 'fixed':
                qty = fixed_plan.get(now, 0)
            else:
                # 50分後の在庫を予測し、完成後2時間分の需要を満たすよう補充
                arrival = now + LEAD
                before_arrival = expected_demand(now, arrival)
                # 予測消費と期限切れを保守的に差し引く
                valid = sum(exp > arrival for exp in inventory)
                valid += sum(q for t, q in pending if t <= arrival)
                projected = max(0, valid - math.ceil(before_arrival))
                # すでに仕込み中のものを二重発注しない
                horizon = min(arrival + 120, END)
                target = math.ceil(1.15 * expected_demand(arrival, horizon))
                qty = max(0, min(CAPACITY - projected, target - projected))
                # 近い将来に仕込んだ在庫が余る場合の過剰補充を抑える
                if expected_demand(arrival, min(arrival + SHELF, END)) < 0.7:
                    qty = 0
            if qty:
                pending.append((now + LEAD, qty))

        hour_index = (now - START) // 60
        lam = HOURLY[hour_index] * STEP / 60 if 0 <= hour_index < len(HOURLY) else 0
        requests = poisson(lam, rng)
        sales = min(requests, len(inventory))
        del inventory[:sales]
        result.sold += sales
        result.missed += requests - sales

    result.wasted += len(inventory)
    return result


def compare(trials=TRIALS):
    for policy, name in [('fixed', '固定製造（7/10/13/16時に各8個）'),
                         ('adaptive', '需要予測による調整製造')]:
        results = [run_day(policy, seed=10000 + i) for i in range(trials)]
        def avg(field):
            return sum(getattr(r, field) for r in results) / trials
        print(f'\n{name} / {trials}日分の仮想実験')
        print(f'  製造: {avg("produced"):.1f}個/日')
        print(f'  販売: {avg("sold"):.1f}個/日')
        print(f'  廃棄: {avg("wasted"):.1f}個/日')
        print(f'  欠品による販売機会損失: {avg("missed"):.1f}個/日')
        print(f'  容量超過で陳列不可: {avg("rejected"):.1f}個/日')


if __name__ == '__main__':
    compare()
    print('\n注意: 需要、製造ルール、費用は仮定。最適性を証明するモデルではありません。')
