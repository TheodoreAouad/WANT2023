def html_template():
      # <link href="/hdd/aouadt/these/projets/3d_segm/deep_morpho/save_results_template/html_template.css" rel="stylesheet">
    return """
    <!DOCTYPE html>
    <html>
      <style>{css_file}</style>
      <head>
        <title>{title}</title>
      </head>
      <body>
        <h2>Global Args</h2>
        <p>{global_args}</p>
        <h2>Changing args</h2>
        <p>{changing_args}</p>
        <h2>Summary</h2>
        <p>{summary}</p>
        <h2>Boxplot</h2>
        <p>{boxplot}</p>
        <h2>Table</h2>
        <p>{table}</p>
        <h2>Results</h2>
        <span>{results}</span>
      </body>
    </html>
    """
