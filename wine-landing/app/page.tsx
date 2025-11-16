import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export default function WineQualityPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      {/* Hero Section */}
      <section className="px-6 py-20 md:py-32">
        <div className="mx-auto max-w-5xl text-center">
          <h1 className="text-5xl font-bold tracking-tight text-foreground md:text-7xl text-balance">
            Wine Quality Forecasting
          </h1>
          <p className="mx-auto mt-6 max-w-3xl text-lg text-muted-foreground md:text-xl text-pretty">
            Predicting French wine vintage quality using 10+ years of weather
            data, deep learning models, and geospatial preprocessing.
          </p>
          <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button
              asChild
              size="lg"
              className="w-full rounded-full px-8 py-6 text-base sm:w-auto"
            >
              <a href="docs/report/wine_quality_report.pdf">📄 Read Report</a>
            </Button>
            <Button
              asChild
              size="lg"
              variant="outline"
              className="w-full rounded-full px-8 py-6 text-base sm:w-auto"
            >
              <a href="/wine_map.html">🍷 Interactive Wine Map</a>
            </Button>
            <Button
              asChild
              size="lg"
              variant="outline"
              className="w-full rounded-full px-8 py-6 text-base sm:w-auto"
            >
              <a
                href="https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes/"
                target="_blank"
                rel="noopener noreferrer"
              >
                🌦 Weather Dataset (Météo-France)
              </a>
            </Button>
          </div>
        </div>
      </section>

      {/* Recruiter Summary Section */}
      <section className="px-6 py-16">
        <div className="mx-auto max-w-4xl">
          <Card className="rounded-3xl p-8 shadow-lg md:p-12">
            <h2 className="text-3xl font-bold text-foreground">
              Why This Project Matters
            </h2>
            <div className="mt-6 space-y-4 text-muted-foreground">
              <p className="text-lg leading-relaxed">
                This project demonstrates <strong>end-to-end ownership</strong>{" "}
                of a complete machine learning pipeline, from raw data
                acquisition to model deployment and visualization.
              </p>
              <p className="text-lg leading-relaxed">
                It showcases expertise in{" "}
                <strong>data engineering, scraping, and merging</strong> complex
                datasets, along with <strong>geospatial processing</strong> to
                link wine regions with weather stations.
              </p>
              <p className="text-lg leading-relaxed">
                The implementation of{" "}
                <strong>deep learning on tabular data</strong> using modern
                architectures (MLP, FT-Transformer, TabNet) highlights practical
                ML engineering skills directly applicable to real-world
                problems.
              </p>
              <p className="text-lg leading-relaxed">
                This work is highly relevant for roles in{" "}
                <strong>AI, ML Engineering, Data Science, and Big Tech</strong>,
                demonstrating the ability to tackle ambiguous problems and
                deliver production-ready solutions.
              </p>
            </div>
          </Card>
        </div>
      </section>

      {/* Skills Demonstrated Section */}
      <section className="px-6 py-16">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-12 text-center text-3xl font-bold text-foreground md:text-4xl">
            Skills Demonstrated
          </h2>
          <div className="grid gap-6 md:grid-cols-3">
            <Card className="rounded-2xl border border-border p-6 shadow-md">
              <h3 className="text-xl font-bold text-foreground">
                Machine Learning
              </h3>
              <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                <li>• MLP, FT-Transformer, TabNet architectures</li>
                <li>• Cross-validation and hyperparameter tuning</li>
                <li>• Classification metrics and evaluation</li>
                <li>• Model interpretation and feature importance</li>
              </ul>
            </Card>

            <Card className="rounded-2xl border border-border p-6 shadow-md">
              <h3 className="text-xl font-bold text-foreground">
                Data Engineering
              </h3>
              <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                <li>• Polars and Pandas for data manipulation</li>
                <li>• Feature engineering and selection</li>
                <li>• Time-series data processing</li>
                <li>• Data cleaning and normalization</li>
              </ul>
            </Card>

            <Card className="rounded-2xl border border-border p-6 shadow-md">
              <h3 className="text-xl font-bold text-foreground">
                Scraping & Data Collection
              </h3>
              <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                <li>• Web scraping from Vivino platform</li>
                <li>• API integration with Météo-France</li>
                <li>• Automated data pipeline construction</li>
                <li>• Robust error handling and retry logic</li>
              </ul>
            </Card>

            <Card className="rounded-2xl border border-border p-6 shadow-md md:col-span-2">
              <h3 className="text-xl font-bold text-foreground">
                Geospatial Processing
              </h3>
              <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                <li>• AOC wine region coordinate mapping</li>
                <li>• Nearest weather station identification</li>
                <li>• Fuzzy matching for location data</li>
                <li>• Spatial joins and distance calculations</li>
              </ul>
            </Card>

            <Card className="rounded-2xl border border-border p-6 shadow-md">
              <h3 className="text-xl font-bold text-foreground">
                Visualization
              </h3>
              <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                <li>• Interactive Plotly maps</li>
                <li>• Data analytics dashboards</li>
                <li>• Model performance visualization</li>
                <li>• Feature correlation heatmaps</li>
              </ul>
            </Card>
          </div>
        </div>
      </section>

      {/* Model Architecture Section */}
      <section className="px-6 py-16">
        <div className="mx-auto max-w-5xl">
          <h2 className="mb-8 text-center text-3xl font-bold text-foreground md:text-4xl">
            Model Architecture Preview
          </h2>
          <div className="rounded-3xl border border-border bg-white p-8 shadow-lg">
            <img
              src="/docs/model_architecture.jpg"
              alt="Model Architecture Diagram"
              className="mx-auto w-full"
            />
            <p className="mt-6 text-center text-sm text-muted-foreground leading-relaxed">
              High-level architecture: categorical embeddings + numeric climate
              features → Feature Builder → MLP or FT-Transformer → quality class
              prediction.
            </p>
          </div>
        </div>
      </section>

      {/* End-to-End Pipeline Section */}
      <section className="px-6 py-16 pb-32">
        <div className="mx-auto max-w-4xl">
          <h2 className="mb-12 text-center text-3xl font-bold text-foreground md:text-4xl">
            End-to-End Pipeline
          </h2>
          <div className="space-y-6">
            {[
              {
                number: "01",
                title: "Vivino Scraping",
                description:
                  "Extract wine ratings, vintage years, and region data from the Vivino platform.",
              },
              {
                number: "02",
                title: "AOC Fuzzy Matching",
                description:
                  "Match wine regions to official AOC (Appellation d'Origine Contrôlée) designations using fuzzy string matching.",
              },
              {
                number: "03",
                title: "Weather Station Cleaning",
                description:
                  "Clean and standardize Météo-France weather station data, handling missing values and outliers.",
              },
              {
                number: "04",
                title: "Climate Feature Engineering",
                description:
                  "Create meaningful climate features including temperature, precipitation, growing degree days, and seasonal aggregations.",
              },
              {
                number: "05",
                title: "Wine–Weather Integration",
                description:
                  "Merge wine vintage data with corresponding weather patterns using geospatial joins and temporal alignment.",
              },
              {
                number: "06",
                title: "Deep Model Training",
                description:
                  "Train multiple deep learning architectures (MLP, FT-Transformer, TabNet) with cross-validation.",
              },
              {
                number: "07",
                title: "Evaluation & Visualization",
                description:
                  "Assess model performance with classification metrics and create interactive visualizations for insights.",
              },
            ].map((step) => (
              <div
                key={step.number}
                className="flex gap-6 rounded-2xl border border-border bg-card p-6 shadow-sm transition-shadow hover:shadow-md"
              >
                <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full bg-primary text-xl font-bold text-primary-foreground">
                  {step.number}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-foreground">
                    {step.title}
                  </h3>
                  <p className="mt-2 text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
