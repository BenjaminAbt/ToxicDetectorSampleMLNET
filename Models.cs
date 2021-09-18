// Copyright © Benjamin Abt 2021, all rights reserved

using CsvHelper.Configuration.Attributes;
using Microsoft.ML.Data;

namespace ToxicSentimentSample;

public class TrainInput
{
    [Index(0)]
    public string CommentId { get; set; } = null!;

    [Index(1)]
    public string Text { get; set; } = null!;

    [Index(2)]
    public bool IsToxic { get; set; }

    [Index(3)]
    public bool IsSevereToxic { get; set; }

    [Index(4)]
    public bool IsObscene { get; set; }

    [Index(5)]
    public bool IsThreat { get; set; }

    [Index(6)]
    public bool IsInsult { get; set; }

    [Index(7)]
    public bool IsIdentityHate { get; set; }
}


public class TextInput
{
    public string Text { get; set; } = null!;
}

public class TextIntentBinaryPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }

    [ColumnName("Score")]
    public float Score { get; set; }
}
