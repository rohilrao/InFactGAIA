# @title
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InFact Analysis: {{ hypothesis }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            line-height: 1.5;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            color: #1a1a1a;
        }

        .card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .hypothesis {
            font-size: 1.2rem;
            color: #4a5568;
            margin: 1rem 0;
            padding: 1rem;
            background: #f7fafc;
            border-radius: 6px;
        }

        .final-assessment {
            text-align: center;
            padding: 2rem;
            background: #ebf8ff;
        }

        .probability {
            font-size: 3rem;
            font-weight: bold;
            color: #2b6cb0;
        }

        .uncertainty {
            font-size: 1.2rem;
            color: #4a5568;
            margin-top: 0.5rem;
        }

        .evidence-point {
            border-left: 4px solid #4299e1;
            padding-left: 1rem;
            margin-bottom: 2rem;
        }

        .evidence-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }

        .stat-card {
            background: #f7fafc;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2b6cb0;
        }

        .stat-label {
            font-size: 0.875rem;
            color: #4a5568;
            margin-top: 0.25rem;
        }

        .chart-container {
            height: 400px;
            margin: 2rem 0;
        }

        .confidence-high { color: #047857; }
        .confidence-medium { color: #b45309; }
        .confidence-low { color: #dc2626; }

        details {
            margin-top: 1rem;
        }

        summary {
            cursor: pointer;
            color: #2b6cb0;
            font-weight: 500;
        }

        pre {
            background: #f7fafc;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>Evidence Analysis</h1>
        <div class="hypothesis">{{ hypothesis }}</div>
    </header>

    <main>
        <section class="card final-assessment">
            <h2>Current Assessment</h2>
            <div class="probability">
                {{ "%.1f"|format(final_probability * 100) }}%
            </div>
            <div class="uncertainty">
                ({{ "%.1f"|format(ci_low * 100) }}%, {{ "%.1f"|format(ci_high * 100) }}%)
            </div>
            <div class="interpretation">
                {% if final_probability > 0.99 %}Virtually Certain
                {% elif final_probability > 0.95 %}Extremely Likely
                {% elif final_probability > 0.90 %}Very Likely
                {% elif final_probability > 0.66 %}Likely
                {% elif final_probability > 0.33 %}Uncertain
                {% elif final_probability > 0.10 %}Unlikely
                {% elif final_probability > 0.05 %}Very Unlikely
                {% elif final_probability > 0.01 %}Extremely Unlikely
                {% else %}Virtually Impossible{% endif %}
            </div>
        </section>

        <section class="card">
            <h2>Belief Evolution</h2>
            <div class="chart-container">
                <canvas id="beliefChart"></canvas>
            </div>
        </section>

        <section class="card">
            <h2>Evidence Analysis</h2>
            {% for point in evidence_points %}
            <div class="evidence-point">
                <h3>Evidence {{ loop.index }}: {{ point.file }}</h3>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.1f"|format(point.prior_prob * 100) }}%</div>
                        <div class="stat-label">Prior Probability</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.1f"|format(point.likelihood_ratio) }}×</div>
                        <div class="stat-label">Likelihood Ratio</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.1f"|format(point.posterior * 100) }}%</div>
                        <div class="stat-label">Posterior Probability</div>
                    </div>
                </div>

                <div class="evidence-grid">
                    <div>
                        <h4>Confidence Assessment</h4>
                        <div class="confidence-score
                            {% if point.confidence_assessment.confidence_score > 0.7 %}confidence-high
                            {% elif point.confidence_assessment.confidence_score > 0.4 %}confidence-medium
                            {% else %}confidence-low{% endif %}">
                            {{ "%.0f"|format(point.confidence_assessment.confidence_score * 100) }}% Confidence
                        </div>

                        {% if point.confidence_assessment.key_strengths %}
                        <h5>Key Strengths</h5>
                        <ul>
                            {% for strength in point.confidence_assessment.key_strengths %}
                            <li>{{ strength }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}

                        {% if point.confidence_assessment.key_limitations %}
                        <h5>Key Limitations</h5>
                        <ul>
                            {% for limitation in point.confidence_assessment.key_limitations %}
                            <li>{{ limitation }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                    </div>
                </div>

                <details>
                    <summary>Analysis Details</summary>
                    <div class="rationale">
                        {% if point.analysis_rationale %}
                        <pre><code>{{ point.analysis_rationale }}</code></pre>
                        {% else %}
                        <p>No detailed analysis rationale available.</p>
                        {% endif %}
                    </div>
                </details>
            </div>
            {% endfor %}
        </section>
    </main>

    <script>
        const ctx = document.getElementById('beliefChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Prior'].concat({{ evidence_points|map(attribute='file')|list|tojson }}),
                datasets: [{
                    label: 'Belief Probability',
                    data: [{{ prior_probability }}].concat({{ evidence_points|map(attribute='posterior')|list|tojson }}),
                    borderColor: '#2b6cb0',
                    backgroundColor: 'rgba(43, 108, 176, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return (context.raw * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""