#!/usr/bin/env python3
"""
Simplified Enhanced Metrics Analysis for Grok-3 Simulations
"""

import json
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def analyze_simulation_file(filepath: str) -> Dict[str, Any]:
    """Analyze a single simulation file with enhanced metrics"""

    with open(filepath, 'r') as f:
        data = json.load(f)

    simulations = data['simulations']
    enhanced_results = []

    for sim in simulations:
        # Extract basic metrics from existing data
        reward = sim['reward_info']['reward']
        messages = sim.get('messages', [])

        # Calculate enhanced metrics
        metrics = calculate_enhanced_metrics(messages, reward, sim)
        enhanced_results.append(metrics)

    # Generate summary
    summary = generate_summary_insights(enhanced_results)

    return {
        'domain_analysis': enhanced_results,
        'summary_insights': summary,
        'total_simulations': len(simulations)
    }


def calculate_enhanced_metrics(messages: List[Dict], reward: float, simulation: Dict) -> Dict[str, Any]:
    """Calculate enhanced metrics for a single simulation"""

    # Basic execution metrics
    execution_score = reward  # Use existing reward as base execution score

    # Communication quality assessment
    agent_messages = [msg for msg in messages if msg.get('role') == 'assistant']
    communication_score = assess_communication_quality(agent_messages)

    # Technical accuracy (simplified)
    technical_score = assess_technical_accuracy(messages)

    # Efficiency metrics
    efficiency_score = assess_efficiency(messages, reward)

    # Overall composite score
    overall_score = (
        execution_score * 0.35 +
        communication_score * 0.25 +
        technical_score * 0.25 +
        efficiency_score * 0.15
    )

    # Failure analysis
    failure_analysis = analyze_failure_pattern(simulation, messages, reward)

    return {
        'execution_score': execution_score,
        'communication_score': communication_score,
        'technical_score': technical_score,
        'efficiency_score': efficiency_score,
        'overall_score': overall_score,
        'failure_analysis': failure_analysis,
        'message_count': len(messages),
        'agent_message_count': len(agent_messages)
    }


def assess_communication_quality(agent_messages: List[Dict]) -> float:
    """Assess communication quality based on agent messages"""
    if not agent_messages:
        return 0.0

    quality_score = 0.0
    total_messages = len(agent_messages)

    for message in agent_messages:
        content = message.get('content', '').lower()

        # Positive indicators
        if any(phrase in content for phrase in ['please', 'let me help', 'i understand', 'can you']):
            quality_score += 1.0
        elif any(phrase in content for phrase in ['step by step', 'first', 'next', 'verify']):
            quality_score += 0.8
        elif len(content) > 50:  # Substantive response
            quality_score += 0.5
        else:
            quality_score += 0.3

    return min(1.0, quality_score / total_messages)


def assess_technical_accuracy(messages: List[Dict]) -> float:
    """Assess technical accuracy based on conversation content"""
    # Simplified assessment based on technical terms and tool usage
    technical_terms = ['check', 'verify', 'enable', 'disable', 'toggle', 'settings', 'network', 'data']
    tool_calls = 0
    technical_content = 0

    for message in messages:
        content = message.get('content', '').lower()
        if any(term in content for term in technical_terms):
            technical_content += 1

        if 'tool_calls' in message or 'function_call' in message:
            tool_calls += 1

    # Base technical score on presence of technical language and tool usage
    technical_score = min(1.0, (technical_content * 0.1 + tool_calls * 0.2))
    return max(0.6, technical_score)  # Minimum baseline technical competency


def assess_efficiency(messages: List[Dict], reward: float) -> float:
    """Assess conversation efficiency"""
    message_count = len(messages)

    # Efficiency decreases with message count, but rewards successful outcomes
    if reward > 0:
        # Successful outcomes - efficiency based on brevity
        if message_count <= 10:
            efficiency = 1.0
        elif message_count <= 20:
            efficiency = 0.8
        elif message_count <= 30:
            efficiency = 0.6
        else:
            efficiency = 0.4
    else:
        # Failed outcomes - lower efficiency
        if message_count <= 5:
            efficiency = 0.3  # Too short, likely premature termination
        elif message_count <= 15:
            efficiency = 0.5
        else:
            efficiency = 0.2  # Long and unsuccessful

    return efficiency


def analyze_failure_pattern(simulation: Dict, messages: List[Dict], reward: float) -> Dict[str, Any]:
    """Analyze failure patterns"""
    if reward > 0:
        return {'failure_type': None, 'failure_reason': 'Success'}

    # Analyze termination reason
    termination_reason = simulation.get('termination_reason', 'unknown')
    message_count = len(messages)
    duration = simulation.get('duration', 0)

    # Categorize failure type
    if termination_reason == 'user_stop':
        if message_count < 5:
            failure_type = 'premature_termination'
            failure_reason = 'Conversation ended too early'
        elif message_count > 30:
            failure_type = 'extended_failure'
            failure_reason = 'Long conversation without resolution'
        else:
            failure_type = 'execution_timing'
            failure_reason = 'User stopped despite ongoing conversation'
    elif termination_reason == 'max_turns':
        failure_type = 'timeout_failure'
        failure_reason = 'Reached maximum conversation length'
    else:
        failure_type = 'unknown_failure'
        failure_reason = f'Terminated due to: {termination_reason}'

    return {
        'failure_type': failure_type,
        'failure_reason': failure_reason,
        'termination_reason': termination_reason,
        'message_count': message_count,
        'duration': duration
    }


def generate_summary_insights(enhanced_results: List[Dict]) -> Dict[str, Any]:
    """Generate summary insights across all simulations"""

    if not enhanced_results:
        return {}

    # Calculate averages
    execution_scores = [r['execution_score'] for r in enhanced_results]
    communication_scores = [r['communication_score'] for r in enhanced_results]
    technical_scores = [r['technical_score'] for r in enhanced_results]
    efficiency_scores = [r['efficiency_score'] for r in enhanced_results]
    overall_scores = [r['overall_score'] for r in enhanced_results]

    # Performance distribution
    excellent = sum(1 for s in overall_scores if s >= 0.8)
    good = sum(1 for s in overall_scores if 0.6 <= s < 0.8)
    acceptable = sum(1 for s in overall_scores if 0.4 <= s < 0.6)
    poor = sum(1 for s in overall_scores if s < 0.4)

    # Failure type analysis
    failure_types = defaultdict(int)
    for result in enhanced_results:
        failure_type = result['failure_analysis']['failure_type']
        if failure_type:
            failure_types[failure_type] += 1

    # Message count statistics
    message_counts = [r['message_count'] for r in enhanced_results]

    return {
        'total_simulations': len(enhanced_results),
        'average_execution_score': np.mean(execution_scores),
        'average_communication_score': np.mean(communication_scores),
        'average_technical_score': np.mean(technical_scores),
        'average_efficiency_score': np.mean(efficiency_scores),
        'average_overall_score': np.mean(overall_scores),
        'performance_distribution': {
            'excellent': excellent,
            'good': good,
            'acceptable': acceptable,
            'poor': poor
        },
        'failure_type_distribution': dict(failure_types),
        'conversation_stats': {
            'average_message_count': np.mean(message_counts),
            'min_message_count': min(message_counts),
            'max_message_count': max(message_counts)
        },
        'success_rate': sum(1 for s in execution_scores if s > 0) / len(execution_scores)
    }


def main():
    """Analyze all three simulation files"""

    simulation_files = [
        ('data/simulations/2025-09-25T22:23:36.445085_retail_llm_agent_grok-3_user_simulator_grok-3.json', 'Retail'),
        ('data/simulations/2025-09-25T22:29:52.595166_airline_llm_agent_grok-3_user_simulator_grok-3.json', 'Airline'),
        ('data/simulations/2025-09-25T22:48:14.059148_telecom_llm_agent_grok-3_user_simulator_grok-3.json', 'Telecom')
    ]

    print('=' * 60)
    print('ENHANCED METRICS ANALYSIS FOR GROK-3 SIMULATIONS')
    print('=' * 60)
    print()

    all_results = {}

    for filepath, domain in simulation_files:
        print(f'--- {domain.upper()} DOMAIN ANALYSIS ---')
        try:
            results = analyze_simulation_file(filepath)
            summary = results['summary_insights']
            all_results[domain] = results

            print(f'Total simulations: {summary["total_simulations"]}')
            print(f'Success rate: {summary["success_rate"]:.1%}')
            print(f'Average overall score: {summary["average_overall_score"]:.3f}')
            print()

            print('Component Scores:')
            print(f'  Execution: {summary["average_execution_score"]:.3f}')
            print(f'  Communication: {summary["average_communication_score"]:.3f}')
            print(f'  Technical: {summary["average_technical_score"]:.3f}')
            print(f'  Efficiency: {summary["average_efficiency_score"]:.3f}')
            print()

            print('Performance Distribution:')
            dist = summary['performance_distribution']
            for tier, count in dist.items():
                print(f'  {tier.capitalize()}: {count} ({count/summary["total_simulations"]:.1%})')
            print()

            print('Conversation Statistics:')
            stats = summary['conversation_stats']
            print(f'  Average messages: {stats["average_message_count"]:.1f}')
            print(f'  Range: {stats["min_message_count"]} - {stats["max_message_count"]}')
            print()

            if summary['failure_type_distribution']:
                print('Failure Types:')
                for failure_type, count in summary['failure_type_distribution'].items():
                    print(f'  {failure_type}: {count}')
                print()

        except Exception as e:
            print(f'Error analyzing {domain}: {e}')
            print()

        print('-' * 50)
        print()

    # Cross-domain comparison
    print('CROSS-DOMAIN COMPARISON')
    print('=' * 30)

    if len(all_results) >= 2:
        domains = list(all_results.keys())
        print('Overall Scores:')
        for domain in domains:
            score = all_results[domain]['summary_insights']['average_overall_score']
            print(f'  {domain}: {score:.3f}')
        print()

        print('Key Insights:')
        # Find best and worst performing domains
        scores = [(domain, all_results[domain]['summary_insights']['average_overall_score'])
                 for domain in domains]
        scores.sort(key=lambda x: x[1], reverse=True)

        print(f'• Best performing domain: {scores[0][0]} ({scores[0][1]:.3f})')
        print(f'• Most challenging domain: {scores[-1][0]} ({scores[-1][1]:.3f})')

        # Efficiency analysis
        efficiency_scores = [(domain, all_results[domain]['summary_insights']['average_efficiency_score'])
                           for domain in domains]
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        print(f'• Most efficient domain: {efficiency_scores[0][0]} ({efficiency_scores[0][1]:.3f})')
        print(f'• Least efficient domain: {efficiency_scores[-1][0]} ({efficiency_scores[-1][1]:.3f})')


if __name__ == "__main__":
    main()