import pandas as pd
from data_processor import DataProcessor
from factor_generator import FactorGenerator
from factor_evaluator import FactorEvaluator
from factor_selector import FactorSelector
from backtester import Backtester
from config import *
from utils import timer

@timer
def main():
    # 1. 数据准备
    print("Step 1: Data Processing...")
    processor = DataProcessor()
    processed_data = processor.process_pipeline()
    processor.save_processed_data()

    # 2. 因子生成
    print("\nStep 2: Factor Generation...")
    generator = FactorGenerator(processed_data)
    factors = generator.generate_all_factors()
    generator.save_factors()

    # 合并所有因子
    all_factors = pd.concat([factors['technical'], factors['cross_section'], factors['interactions']], axis=1)

    # 3. 因子评估
    print("\nStep 3: Factor Evaluation...")
    evaluator = FactorEvaluator(factors, processed_data['returns'])
    evaluation_results = evaluator.evaluate_all_factors(processed_data['price'])
    evaluator.save_evaluation_results()

    # 4. 因子选择
    print("\nStep 4: Factor Selection...")
    selector = FactorSelector(all_factors, processed_data['returns'])
    selected_factors = selector.select_all_methods()
    consensus_factors = selector.get_consensus_factors()
    selector.save_selected_factors()

    print(f"\nSelected {len(consensus_factors)} consensus factors:")
    print(consensus_factors.tolist())

    # 5. 组合回测
    print("\nStep 5: Portfolio Backtesting...")
    backtester = Backtester(all_factors[consensus_factors], processed_data['returns'], processed_data['price'])
    backtest_results = backtester.backtest_all_factors()
    backtester.save_backtest_results()

    print("\nFactor Mining Process Completed!")

if __name__ == "__main__":
    main()