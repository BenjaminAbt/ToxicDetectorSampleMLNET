// Copyright © Benjamin Abt 2021, all rights reserved

using Microsoft.ML;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.IO;
using ToxicSentimentSample;

// settings
string trainDataFile = Path.Combine(Environment.CurrentDirectory, "train.csv");

// Context
MLContext mlContext = new();

AnsiConsole.Render(new FigletText("ML.NET Toxic Sentiment Sample").LeftAligned().Color(Color.Red));

// create ML context
AnsiConsole.MarkupLine("=> create model context...");

// read train data
AnsiConsole.MarkupLine("=> loading test data...");
List<TrainInput> trainData = DataReader.ReadTrainData(trainDataFile);
if(trainData.Count == 0)
{
    AnsiConsole.MarkupLine("[red]!!! No testdata found. Have you downloaded the train.csv?[/]");
    return;
}

// load train data
DataOperationsCatalog.TrainTestData dataView;
{
    AnsiConsole.MarkupLine("=> parsing test data...");
    // on load from enumerable, the property information (and attribute settings) are used for headers and features.
    IDataView data = mlContext.Data.LoadFromEnumerable(trainData);

    AnsiConsole.MarkupLine("=> create model partitions...");
    dataView = mlContext.Data.TrainTestSplit(data, testFraction: 0.2); // create partitions, we use 20% testing and 80% training
}
// setup model options
AnsiConsole.MarkupLine("=> create binary classification pipeline...");
var textPipeline = BinaryClassification.CreatePipeline(mlContext);

// train model
AnsiConsole.MarkupLine("=> model training...");
ITransformer trainedModel = textPipeline.Fit(dataView.TrainSet);

// validate model
AnsiConsole.MarkupLine("=> model validation...");
IDataView predictions = trainedModel.Transform(dataView.TestSet);
var metrics = BinaryClassification.Validate(mlContext, predictions);

// report the results
BinaryClassification.PrintMetrics(metrics);
// user data
PredictionEngine<TextInput, TextIntentBinaryPrediction> predictionEngine =
    mlContext.Model.CreatePredictionEngine<TextInput, TextIntentBinaryPrediction>(trainedModel);

while (true)
{
    AnsiConsole.MarkupLine("[blue]--------------------------------------------------[/]");
    string? predictInput = AnsiConsole.Ask<string?>("Text Input: ");
    if (string.IsNullOrEmpty(predictInput))
    {
        break;
    }

    TextInput input = new() { Text = predictInput };
    TextIntentBinaryPrediction prediction = predictionEngine.Predict(input);

    AnsiConsole.MarkupLine(Environment.NewLine + "[green]Predict Results:[/]");

    Table prt = new();
    {
        prt.AddColumn(new TableColumn("Name").RightAligned());
        prt.AddColumn("Value");
    }


    prt.AddRow("Text", input.Text)
                .AddRow("Prediction", prediction.Prediction ? "Toxic :-(" : "Friendly :-)")
                .AddRow("Probability", $"{prediction.Probability:P2}")
                .AddRow("Score", $"{prediction.Score}");

    AnsiConsole.Render(prt);
}
