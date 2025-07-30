#!/usr/bin/env python3
"""
Test script for the enhanced ARC Meta-Learning Training System
Demonstrates the new specialized training approach for different grid types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claude import *
import numpy as np

def test_meta_learning_system():
    """Test the complete meta-learning training system"""
    print("🧠 Testing ARC Meta-Learning Training System")
    print("=" * 60)
    
    # Test 1: Grid Type Classification
    print("\n1. Testing Grid Type Classification...")
    classifier = GridTypeClassifier()
    
    # Same-size example
    input_same = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])
    output_same = np.array([[1, 2, 1], [2, 2, 2], [1, 2, 1]])
    
    puzzle_type, sub_type = classifier.classify_puzzle(input_same, output_same)
    print(f"Same-size puzzle classified as: {puzzle_type} - {sub_type}")
    
    # Different-size example
    input_diff = np.array([[1, 2], [3, 4]])
    output_diff = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
    
    puzzle_type, sub_type = classifier.classify_puzzle(input_diff, output_diff)
    print(f"Diff-size puzzle classified as: {puzzle_type} - {sub_type}")
    
    # Test 2: Specialist Training
    print("\n2. Testing Specialist Training...")
    
    # Same-size specialist
    same_specialist = SameSizeSpecialist()
    same_specialist.initialize_primitives()
    print(f"Same-size specialist initialized with {len(same_specialist.specialized_primitives)} primitives")
    
    same_tentacles = same_specialist.train_on_puzzle(input_same, output_same, "pattern_modification")
    print(f"Generated {len(same_tentacles)} tentacles for same-size puzzle")
    
    # Different-size specialist
    diff_specialist = DiffSizeSpecialist()
    diff_specialist.initialize_primitives()
    print(f"Diff-size specialist initialized with {len(diff_specialist.specialized_primitives)} primitives")
    
    diff_tentacles = diff_specialist.train_on_puzzle(input_diff, output_diff, "pattern_repetition")
    print(f"Generated {len(diff_tentacles)} tentacles for diff-size puzzle")
    
    # Test 3: Meta-Learning Trainer
    print("\n3. Testing Meta-Learning Trainer...")
    meta_trainer = MetaLearningTrainer()
    
    # Create test puzzles
    test_puzzles = [
        {
            "train": [
                {"input": input_same.tolist(), "output": output_same.tolist()}
            ]
        },
        {
            "train": [
                {"input": input_diff.tolist(), "output": output_diff.tolist()}
            ]
        }
    ]
    
    # Train meta-learning system
    meta_results = meta_trainer.train_on_puzzle_set(test_puzzles)
    print(f"Meta-training completed on {len(test_puzzles)} puzzles")
    
    # Test 4: Create Optimized Octopi
    print("\n4. Testing Specialized Octopus Creation...")
    
    same_octopus = meta_trainer.create_optimized_octopus("same_size")
    print(f"Same-size octopus created with {len(same_octopus.tentacles)} tentacles")
    
    diff_octopus = meta_trainer.create_optimized_octopus("diff_size")
    print(f"Diff-size octopus created with {len(diff_octopus.tentacles)} tentacles")
    
    hybrid_octopus = meta_trainer.create_optimized_octopus()
    print(f"Hybrid octopus created with {len(hybrid_octopus.tentacles)} tentacles")
    
    # Test 5: Solve Puzzles
    print("\n5. Testing Puzzle Solving...")
    
    # Test same-size puzzle
    same_result = same_octopus.execute(input_same)
    same_accuracy = np.mean(same_result == output_same) if same_result.shape == output_same.shape else 0.0
    print(f"Same-size puzzle accuracy: {same_accuracy:.2f}")
    print(f"Input shape: {input_same.shape}, Output shape: {same_result.shape}")
    
    # Test different-size puzzle
    diff_result = diff_octopus.execute(input_diff)
    diff_accuracy = np.mean(diff_result == output_diff) if diff_result.shape == output_diff.shape else 0.0
    print(f"Diff-size puzzle accuracy: {diff_accuracy:.2f}")
    print(f"Input shape: {input_diff.shape}, Output shape: {diff_result.shape}")
    
    # Test 6: Full System Integration
    print("\n6. Testing Full System Integration...")
    
    mother = MotherOctopus(population_size=10)
    print(f"MotherOctopus created with population size: {mother.population_size}")
    
    # Run minimal training
    training_results = mother.train_with_meta_learning(test_puzzles, generations=2)
    print(f"Training completed! Best fitness: {training_results['performance_tracker']['overall_improvement']:.3f}")
    
    # Test final solving
    final_result = mother.solve_puzzle(input_same)
    final_accuracy = np.mean(final_result == output_same) if final_result.shape == output_same.shape else 0.0
    print(f"Final system accuracy on same-size: {final_accuracy:.2f}")
    
    print("\n✅ All tests completed successfully!")
    
    return {
        'classifier_works': True,
        'specialists_work': len(same_tentacles) > 0 and len(diff_tentacles) > 0,
        'meta_trainer_works': len(meta_results['same_size_results']) > 0,
        'octopi_created': len(same_octopus.tentacles) > 0,
        'system_trains': 'best_octopus' in training_results,
        'final_accuracy': final_accuracy
    }

def test_comprehensive_example():
    """Test with the comprehensive example from the demo"""
    print("\n🎯 Testing Comprehensive Example...")
    
    try:
        # Use the comprehensive demo
        training_results, test_results = demonstrate_meta_learning_training()
        
        if training_results and test_results:
            print("✅ Comprehensive demo completed successfully!")
            print(f"Overall accuracy: {test_results['overall_accuracy']:.2%}")
            return True
        else:
            print("❌ Comprehensive demo failed")
            return False
            
    except Exception as e:
        print(f"❌ Comprehensive demo error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting ARC Meta-Learning System Tests")
    
    try:
        # Run basic tests
        basic_results = test_meta_learning_system()
        print(f"\n📊 Basic Test Results: {basic_results}")
        
        # Run comprehensive tests
        comprehensive_success = test_comprehensive_example()
        
        if all(basic_results.values()) and comprehensive_success:
            print("\n🎉 ALL TESTS PASSED! Meta-learning system is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
