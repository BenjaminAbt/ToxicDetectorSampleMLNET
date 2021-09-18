// Copyright © Benjamin Abt 2021, all rights reserved

using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Spectre.Console;

namespace ToxicSentimentSample
{
    public static class BinaryClassification
    {
        public static EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>>
            CreatePipeline(MLContext mlContext)
        {
            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>> pipeline
                = mlContext.Transforms.Text.FeaturizeText(
                            outputColumnName: "Features",
                            inputColumnName: nameof(TrainInput.Text))

                            // step 2: add a fast tree learner
                            .Append(mlContext.BinaryClassification.Trainers.FastTree(
                                labelColumnName: nameof(TrainInput.IsToxic),
                                featureColumnName: "Features"));

            return pipeline;
        }

        public static CalibratedBinaryClassificationMetrics Validate(MLContext mlContext, IDataView predictions)
            => mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(TrainInput.IsToxic));

        public static void PrintMetrics(CalibratedBinaryClassificationMetrics metrics)
        {
            AnsiConsole.MarkupLine($"=> metrics for binary classification");

            Table trt = new();
            {
                trt.AddColumn(new TableColumn("Name").RightAligned());
                trt.AddColumn("Value");
            }

            trt.AddRow("Accuracy", $"{metrics.Accuracy:P2}");
            trt.AddRow("Area Under Precision Recall Curve", $"{metrics.AreaUnderPrecisionRecallCurve:P2}");
            trt.AddRow("Area Under Roc Curve", $"{metrics.AreaUnderRocCurve:P2}");
            trt.AddRow("F1 Score", $"{metrics.F1Score:P2}");
            trt.AddRow("Log Loss", $"{metrics.LogLoss:0.##}");
            trt.AddRow("Log Loss Reduction", $"{metrics.LogLossReduction:0.##}");
            trt.AddRow("Positive Precision", $"{metrics.PositivePrecision:0.##}");
            trt.AddRow("Positive Recall", $"{metrics.PositiveRecall:0.##}");
            trt.AddRow("Negative Precision", $"{metrics.NegativePrecision:0.##}");
            trt.AddRow("Negative Recall", $"{metrics.NegativeRecall:0.##}");

            AnsiConsole.Render(trt);
        }
    }
}
