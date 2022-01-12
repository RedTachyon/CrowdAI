using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class ColorMap
{
    // Source: https://sashamaps.net/docs/resources/20-colors/
    // private static List<string> _hexes = new()
    // {
    //     "#800000",
    //     "#9a6324",
    //     "#808000",
    //     "#469990",
    //     "#000075",
    //     "#000000",
    //     "#e6194B",
    //     "#f58231",
    //     "#ffe119",
    //     "#bfef45",
    //     "#3cb44b",
    //     "#42d4f4",
    //     "#4363d8",
    //     "#911eb4",
    //     "#f032e6",
    //     "#a9a9a9",
    //     "#fabed4",
    //     "#ffd8b1",
    //     "#fffac8",
    //     "#aaffc3",
    //     "#dcbeff",
    //     "#ffffff",
    // };
    //
    // private static List<Color> _colors = _hexes.Select(Hex).ToList();

    private static List<Color> _colors = new()
    {
        // flounder.com/csharp_color_table.html
        Color.black,
        Color.blue,
        Color.cyan,
        Color.green,
        Color.magenta,
        Color.red,
        Color.yellow,
        new Color(255, 127, 80)/255f, // Coral
        new Color(139, 0, 139)/255f, // DarkMagenta
        new Color(40, 79, 79)/255f, // DarkSlateGrey
        new Color(255, 215, 0)/255f, // Gold
        new Color(255, 182, 193)/255f // LightPink
    };

    private static int numColors = _colors.Count;
  
    public static Color GetColor(int idx)
    {
        return _colors[idx % numColors];
    }

    public static Color Hex(string hex)
    {
        var r = Convert.ToInt32($"0x{hex.Substring(1, 2)}", 16);
        var g = Convert.ToInt32($"0x{hex.Substring(3, 2)}", 16);
        var b = Convert.ToInt32($"0x{hex.Substring(5, 2)}", 16);

        return new Color(r, g, b);
    }
}