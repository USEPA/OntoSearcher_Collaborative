import rdflib
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL
from collections import defaultdict
import textwrap
import re
import os
from datetime import datetime

def get_readable_label(uri, graph=None):
    """
    Create a readable label for a URI, using RDFS label if available
    
    Parameters:
    uri (rdflib.URIRef): The URI to get a label for
    graph (rdflib.Graph, optional): The graph to search for labels
    
    Returns:
    str: A readable label
    """
    if uri is None:
        return "None"
    
    uri_str = str(uri)
    
    # If we have a graph, check for rdfs:label
    if graph is not None:
        for _, _, label in graph.triples((uri, RDFS.label, None)):
            return str(label)
    
    # Try to extract from the URI
    # Check for common namespaces first
    common_namespaces = {
        str(XSD): "xsd:",
        str(RDF): "rdf:",
        str(RDFS): "rdfs:",
        str(OWL): "owl:",
        "http://schema.org/": "schema:",
        "http://xmlns.com/foaf/0.1/": "foaf:",
        "http://purl.org/dc/elements/1.1/": "dc:",
        "http://purl.org/dc/terms/": "dcterms:",
    }
    
    for namespace, prefix in common_namespaces.items():
        if uri_str.startswith(namespace):
            return f"{prefix}{uri_str[len(namespace):]}"
    
    # Try to get a nice label from the URI itself
    if '#' in uri_str:
        local_name = uri_str.split('#')[-1]
    elif '/' in uri_str:
        local_name = uri_str.split('/')[-1]
    else:
        local_name = uri_str
    
    # Convert CamelCase to spaces
    local_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', local_name)
    # Convert underscores to spaces
    local_name = local_name.replace('_', ' ')
    # Capitalize first letter
    if local_name:
        local_name = local_name[0].upper() + local_name[1:]
    
    return local_name

def get_namespace_info(graph):
    """
    Extract and organize namespace information from the graph
    
    Parameters:
    graph (rdflib.Graph): The RDF graph
    
    Returns:
    dict: A dictionary mapping prefix to namespace URI
    """
    namespace_info = {}
    
    for prefix, namespace in graph.namespaces():
        namespace_info[prefix] = str(namespace)
    
    return namespace_info

def get_description(uri, graph):
    """
    Get description, comment or other documentation for a URI
    
    Parameters:
    uri (rdflib.URIRef): The URI to get description for
    graph (rdflib.Graph): The RDF graph
    
    Returns:
    str: A description or empty string if none found
    """
    # Check for rdfs:comment
    for _, _, comment in graph.triples((uri, RDFS.comment, None)):
        return str(comment)
    
    # Check for other common description properties
    description_properties = [
        RDFS.comment,
        URIRef("http://purl.org/dc/elements/1.1/description"),
        URIRef("http://purl.org/dc/terms/description"),
        URIRef("http://schema.org/description"),
        URIRef("http://www.w3.org/2004/02/skos/core#definition")
    ]
    
    for prop in description_properties:
        for _, _, desc in graph.triples((uri, prop, None)):
            return str(desc)
    
    return ""

def get_property_details(prop, graph):
    """
    Get detailed information about a property
    
    Parameters:
    prop (rdflib.URIRef): The property URI
    graph (rdflib.Graph): The RDF graph
    
    Returns:
    dict: A dictionary with property details
    """
    details = {
        'uri': str(prop),
        'label': get_readable_label(prop, graph),
        'description': get_description(prop, graph),
        'domains': [],
        'ranges': [],
        'super_properties': [],
        'type': 'Property'
    }
    
    # Get domains (what classes can have this property)
    for _, _, domain in graph.triples((prop, RDFS.domain, None)):
        details['domains'].append({
            'uri': str(domain),
            'label': get_readable_label(domain, graph)
        })
    
    # Get ranges (what values this property can have)
    for _, _, range_uri in graph.triples((prop, RDFS.range, None)):
        details['ranges'].append({
            'uri': str(range_uri),
            'label': get_readable_label(range_uri, graph)
        })
    
    # Get super-properties
    for _, _, super_prop in graph.triples((prop, RDFS.subPropertyOf, None)):
        details['super_properties'].append({
            'uri': str(super_prop),
            'label': get_readable_label(super_prop, graph)
        })
    
    # Check if it's a datatype property or object property
    is_datatype_prop = False
    for range_info in details['ranges']:
        if range_info['uri'].startswith(str(XSD)):
            is_datatype_prop = True
            break
    
    # Determine property type
    if (prop, RDF.type, OWL.DatatypeProperty) in graph:
        details['type'] = 'Datatype Property'
    elif (prop, RDF.type, OWL.ObjectProperty) in graph:
        details['type'] = 'Object Property'
    elif is_datatype_prop:
        details['type'] = 'Datatype Property'
    else:
        details['type'] = 'Object Property'
    
    return details

def get_class_details(cls, graph):
    """
    Get detailed information about a class
    
    Parameters:
    cls (rdflib.URIRef): The class URI
    graph (rdflib.Graph): The RDF graph
    
    Returns:
    dict: A dictionary with class details
    """
    details = {
        'uri': str(cls),
        'label': get_readable_label(cls, graph),
        'description': get_description(cls, graph),
        'super_classes': [],
        'sub_classes': [],
        'properties': []
    }
    
    # Get super-classes
    for _, _, super_cls in graph.triples((cls, RDFS.subClassOf, None)):
        if isinstance(super_cls, URIRef):
            details['super_classes'].append({
                'uri': str(super_cls),
                'label': get_readable_label(super_cls, graph)
            })
    
    # Get sub-classes
    for sub_cls, _, _ in graph.triples((None, RDFS.subClassOf, cls)):
        if isinstance(sub_cls, URIRef):
            details['sub_classes'].append({
                'uri': str(sub_cls),
                'label': get_readable_label(sub_cls, graph)
            })
    
    # Get properties that have this class as domain
    for prop, _, _ in graph.triples((None, RDFS.domain, cls)):
        if (prop, RDF.type, RDF.Property) in graph or (prop, RDF.type, OWL.DatatypeProperty) in graph or (prop, RDF.type, OWL.ObjectProperty) in graph:
            prop_info = {
                'uri': str(prop),
                'label': get_readable_label(prop, graph),
                'description': get_description(prop, graph)
            }
            
            # Get the range for this property
            ranges = []
            for _, _, range_uri in graph.triples((prop, RDFS.range, None)):
                ranges.append({
                    'uri': str(range_uri),
                    'label': get_readable_label(range_uri, graph)
                })
            
            prop_info['ranges'] = ranges
            details['properties'].append(prop_info)
    
    return details

def generate_schema_documentation(input_file, output_file=None, format='turtle'):
    """
    Generate human-readable documentation from an RDF schema file
    
    Parameters:
    input_file (str): Path to the RDF schema file
    output_file (str, optional): Path to save the documentation (if None, returns the text)
    format (str): Format of the RDF file (e.g., 'turtle', 'xml', 'json-ld')
    
    Returns:
    str: The generated documentation text (if output_file is None)
    """
    # Load the RDF schema
    g = Graph()
    g.parse(input_file, format=format)
    
    # Create lists to store detailed information
    classes = []
    properties = []
    
    # Extract classes
    for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
        if isinstance(s, URIRef):
            classes.append(get_class_details(s, g))
    
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef) and not any(c['uri'] == str(s) for c in classes):
            classes.append(get_class_details(s, g))
    
    # Extract properties
    for s, p, o in g.triples((None, RDF.type, RDF.Property)):
        if isinstance(s, URIRef):
            properties.append(get_property_details(s, g))
    
    for s, p, o in g.triples((None, RDF.type, OWL.DatatypeProperty)):
        if isinstance(s, URIRef) and not any(p['uri'] == str(s) for p in properties):
            properties.append(get_property_details(s, g))
            
    for s, p, o in g.triples((None, RDF.type, OWL.ObjectProperty)):
        if isinstance(s, URIRef) and not any(p['uri'] == str(s) for p in properties):
            properties.append(get_property_details(s, g))
    
    # Extract namespaces
    namespaces = get_namespace_info(g)
    
    # Sort classes and properties alphabetically by label
    classes.sort(key=lambda x: x['label'].lower())
    properties.sort(key=lambda x: x['label'].lower())
    
    # Generate the documentation
    doc = []
    
    # Title and introduction
    doc.append("=" * 80)
    doc.append(" " * 25 + "RDF SCHEMA DOCUMENTATION")
    doc.append("=" * 80)
    doc.append("")
    doc.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.append(f"Source file: {os.path.basename(input_file)}")
    doc.append(f"Contains: {len(classes)} classes and {len(properties)} properties")
    doc.append("")
    
    # Table of Contents
    doc.append("TABLE OF CONTENTS")
    doc.append("=" * 17)
    doc.append("1. NAMESPACES")
    doc.append("2. CLASSES")
    for i, cls in enumerate(classes):
        doc.append(f"   2.{i+1}. {cls['label']}")
    doc.append("3. PROPERTIES")
    for i, prop in enumerate(properties):
        doc.append(f"   3.{i+1}. {prop['label']}")
    doc.append("")
    doc.append("-" * 80)
    doc.append("")
    
    # Namespaces
    doc.append("1. NAMESPACES")
    doc.append("=" * 12)
    doc.append("The following namespaces are used in this schema:")
    doc.append("")
    
    for prefix, uri in sorted(namespaces.items()):
        doc.append(f"Prefix: {prefix}")
        doc.append(f"URI: {uri}")
        doc.append("")
    
    doc.append("-" * 80)
    doc.append("")
    
    # Classes
    doc.append("2. CLASSES")
    doc.append("=" * 10)
    doc.append(f"This schema defines {len(classes)} classes:")
    doc.append("")
    
    for i, cls in enumerate(classes):
        doc.append(f"2.{i+1}. {cls['label']}")
        doc.append("-" * len(f"2.{i+1}. {cls['label']}"))
        doc.append(f"URI: {cls['uri']}")
        
        if cls['description']:
            doc.append("")
            doc.append("Description:")
            doc.extend(textwrap.wrap(cls['description'], width=78))
        
        if cls['super_classes']:
            doc.append("")
            doc.append("Parent classes:")
            for super_cls in cls['super_classes']:
                doc.append(f"- {super_cls['label']} ({super_cls['uri']})")
        
        if cls['sub_classes']:
            doc.append("")
            doc.append("Child classes:")
            for sub_cls in cls['sub_classes']:
                doc.append(f"- {sub_cls['label']} ({sub_cls['uri']})")
        
        if cls['properties']:
            doc.append("")
            doc.append("Properties:")
            for prop in cls['properties']:
                range_str = ""
                if prop['ranges']:
                    range_labels = [r['label'] for r in prop['ranges']]
                    range_str = f" (values: {', '.join(range_labels)})"
                
                doc.append(f"- {prop['label']}{range_str}")
                
                # If property has description, indent and add it
                if prop['description']:
                    wrapped_desc = textwrap.wrap(prop['description'], width=74)
                    doc.extend([f"  {line}" for line in wrapped_desc])
        
        doc.append("")
        doc.append("-" * 80)
        doc.append("")
    
    # Properties
    doc.append("3. PROPERTIES")
    doc.append("=" * 12)
    doc.append(f"This schema defines {len(properties)} properties:")
    doc.append("")
    
    for i, prop in enumerate(properties):
        doc.append(f"3.{i+1}. {prop['label']}")
        doc.append("-" * len(f"3.{i+1}. {prop['label']}"))
        doc.append(f"URI: {prop['uri']}")
        doc.append(f"Type: {prop['type']}")
        
        if prop['description']:
            doc.append("")
            doc.append("Description:")
            doc.extend(textwrap.wrap(prop['description'], width=78))
        
        if prop['domains']:
            doc.append("")
            doc.append("Domain (classes that can have this property):")
            for domain in prop['domains']:
                doc.append(f"- {domain['label']} ({domain['uri']})")
        
        if prop['ranges']:
            doc.append("")
            doc.append("Range (values this property can have):")
            for range_info in prop['ranges']:
                doc.append(f"- {range_info['label']} ({range_info['uri']})")
        
        if prop['super_properties']:
            doc.append("")
            doc.append("Parent properties:")
            for super_prop in prop['super_properties']:
                doc.append(f"- {super_prop['label']} ({super_prop['uri']})")
        
        doc.append("")
        doc.append("-" * 80)
        doc.append("")
    
    # Join all lines into a single string
    doc_text = "\n".join(doc)
    
    # Save to file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(doc_text)
        print(f"Documentation saved to {output_file}")
        return None
    else:
        return doc_text

def generate_schema_markdown(input_file, output_file=None, format='turtle'):
    """
    Generate human-readable markdown documentation from an RDF schema file
    
    Parameters:
    input_file (str): Path to the RDF schema file
    output_file (str, optional): Path to save the documentation (if None, returns the text)
    format (str): Format of the RDF file (e.g., 'turtle', 'xml', 'json-ld')
    
    Returns:
    str: The generated markdown documentation (if output_file is None)
    """
    # Load the RDF schema
    g = Graph()
    g.parse(input_file, format=format)
    
    # Create lists to store detailed information
    classes = []
    properties = []
    
    # Extract classes
    for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
        if isinstance(s, URIRef):
            classes.append(get_class_details(s, g))
    
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef) and not any(c['uri'] == str(s) for c in classes):
            classes.append(get_class_details(s, g))
    
    # Extract properties
    for s, p, o in g.triples((None, RDF.type, RDF.Property)):
        if isinstance(s, URIRef):
            properties.append(get_property_details(s, g))
    
    for s, p, o in g.triples((None, RDF.type, OWL.DatatypeProperty)):
        if isinstance(s, URIRef) and not any(p['uri'] == str(s) for p in properties):
            properties.append(get_property_details(s, g))
            
    for s, p, o in g.triples((None, RDF.type, OWL.ObjectProperty)):
        if isinstance(s, URIRef) and not any(p['uri'] == str(s) for p in properties):
            properties.append(get_property_details(s, g))
    
    # Extract namespaces
    namespaces = get_namespace_info(g)
    
    # Sort classes and properties alphabetically by label
    classes.sort(key=lambda x: x['label'].lower())
    properties.sort(key=lambda x: x['label'].lower())
    
    # Generate the markdown documentation
    doc = []
    
    # Title and introduction
    doc.append("# RDF Schema Documentation")
    doc.append("")
    doc.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    doc.append(f"**Source file:** {os.path.basename(input_file)}  ")
    doc.append(f"**Contains:** {len(classes)} classes and {len(properties)} properties  ")
    doc.append("")
    
    # Table of Contents
    doc.append("## Table of Contents")
    doc.append("")
    doc.append("1. [Namespaces](#1-namespaces)")
    doc.append("2. [Classes](#2-classes)")
    for i, cls in enumerate(classes):
        doc.append(f"   - [{cls['label']}](#2{i+1}-{cls['label'].lower().replace(' ', '-')})")
    doc.append("3. [Properties](#3-properties)")
    for i, prop in enumerate(properties):
        doc.append(f"   - [{prop['label']}](#3{i+1}-{prop['label'].lower().replace(' ', '-')})")
    doc.append("")
    doc.append("---")
    doc.append("")
    
    # Namespaces
    doc.append("## 1. Namespaces")
    doc.append("")
    doc.append("The following namespaces are used in this schema:")
    doc.append("")
    
    doc.append("| Prefix | URI |")
    doc.append("| ------ | --- |")
    for prefix, uri in sorted(namespaces.items()):
        doc.append(f"| {prefix} | {uri} |")
    
    doc.append("")
    doc.append("---")
    doc.append("")
    
    # Classes
    doc.append("## 2. Classes")
    doc.append("")
    doc.append(f"This schema defines {len(classes)} classes:")
    doc.append("")
    
    for i, cls in enumerate(classes):
        doc.append(f"### 2.{i+1}. {cls['label']}")
        doc.append("")
        doc.append(f"**URI:** `{cls['uri']}`")
        
        if cls['description']:
            doc.append("")
            doc.append("**Description:**")
            doc.append(cls['description'])
        
        if cls['super_classes']:
            doc.append("")
            doc.append("**Parent classes:**")
            for super_cls in cls['super_classes']:
                doc.append(f"- {super_cls['label']} (`{super_cls['uri']}`)")
        
        if cls['sub_classes']:
            doc.append("")
            doc.append("**Child classes:**")
            for sub_cls in cls['sub_classes']:
                doc.append(f"- {sub_cls['label']} (`{sub_cls['uri']}`)")
        
        if cls['properties']:
            doc.append("")
            doc.append("**Properties:**")
            doc.append("")
            doc.append("| Property | Values | Description |")
            doc.append("| -------- | ------ | ----------- |")
            
            for prop in cls['properties']:
                range_str = ""
                if prop['ranges']:
                    range_labels = [r['label'] for r in prop['ranges']]
                    range_str = ", ".join(range_labels)
                
                desc = prop['description'] if prop['description'] else ""
                # Ensure markdown table compatibility
                desc = desc.replace("\n", " ").replace("|", "\\|")
                
                doc.append(f"| {prop['label']} | {range_str} | {desc} |")
        
        doc.append("")
        doc.append("---")
        doc.append("")
    
    # Properties
    doc.append("## 3. Properties")
    doc.append("")
    doc.append(f"This schema defines {len(properties)} properties:")
    doc.append("")
    
    for i, prop in enumerate(properties):
        doc.append(f"### 3.{i+1}. {prop['label']}")
        doc.append("")
        doc.append(f"**URI:** `{prop['uri']}`  ")
        doc.append(f"**Type:** {prop['type']}")
        
        if prop['description']:
            doc.append("")
            doc.append("**Description:**")
            doc.append(prop['description'])
        
        if prop['domains']:
            doc.append("")
            doc.append("**Domain (classes that can have this property):**")
            for domain in prop['domains']:
                doc.append(f"- {domain['label']} (`{domain['uri']}`)")
        
        if prop['ranges']:
            doc.append("")
            doc.append("**Range (values this property can have):**")
            for range_info in prop['ranges']:
                doc.append(f"- {range_info['label']} (`{range_info['uri']}`)")
        
        if prop['super_properties']:
            doc.append("")
            doc.append("**Parent properties:**")
            for super_prop in prop['super_properties']:
                doc.append(f"- {super_prop['label']} (`{super_prop['uri']}`)")
        
        doc.append("")
        doc.append("---")
        doc.append("")
    
    # Join all lines into a single string
    doc_text = "\n".join(doc)
    
    # Save to file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(doc_text)
        print(f"Markdown documentation saved to {output_file}")
        return None
    else:
        return doc_text

def generate_schema_html(input_file, output_file=None, format='turtle'):
    """
    Generate human-readable HTML documentation from an RDF schema file
    
    Parameters:
    input_file (str): Path to the RDF schema file
    output_file (str, optional): Path to save the documentation (if None, returns the text)
    format (str): Format of the RDF file (e.g., 'turtle', 'xml', 'json-ld')
    
    Returns:
    str: The generated HTML documentation (if output_file is None)
    """
    # Load the RDF schema
    g = Graph()
    g.parse(input_file, format=format)
    
    # Create lists to store detailed information
    classes = []
    properties = []
    
    # Extract classes
    for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
        if isinstance(s, URIRef):
            classes.append(get_class_details(s, g))
    
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef) and not any(c['uri'] == str(s) for c in classes):
            classes.append(get_class_details(s, g))
    
    # Extract properties
    for s, p, o in g.triples((None, RDF.type, RDF.Property)):
        if isinstance(s, URIRef):
            properties.append(get_property_details(s, g))
    
    for s, p, o in g.triples((None, RDF.type, OWL.DatatypeProperty)):
        if isinstance(s, URIRef) and not any(p['uri'] == str(s) for p in properties):
            properties.append(get_property_details(s, g))
            
    for s, p, o in g.triples((None, RDF.type, OWL.ObjectProperty)):
        if isinstance(s, URIRef) and not any(p['uri'] == str(s) for p in properties):
            properties.append(get_property_details(s, g))
    
    # Extract namespaces
    namespaces = get_namespace_info(g)
    
    # Sort classes and properties alphabetically by label
    classes.sort(key=lambda x: x['label'].lower())
    properties.sort(key=lambda x: x['label'].lower())
    
    # Generate the HTML documentation
    html = []
    
    # HTML header
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("  <meta charset='UTF-8'>")
    html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append(f"  <title>RDF Schema Documentation - {os.path.basename(input_file)}</title>")
    html.append("  <style>")
    html.append("    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }")
    html.append("    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }")
    html.append("    h2 { color: #2980b9; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }")
    html.append("    h3 { color: #3498db; margin-top: 25px; }")
    html.append("    a { color: #3498db; text-decoration: none; }")
    html.append("    a:hover { text-decoration: underline; }")
    html.append("    .container { max-width: 1200px; margin: 0 auto; }")
    html.append("    .info { color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }")
    html.append("    .uri { font-family: monospace; background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }")
    html.append("    table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html.append("    th { background-color: #f2f2f2; }")
    html.append("    tr:nth-child(even) { background-color: #f9f9f9; }")
    html.append("    .toc { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }")
    html.append("    .toc ul { list-style-type: none; padding-left: 15px; }")
    html.append("    .toc li { margin: 5px 0; }")
    html.append("    .back-to-top { position: fixed; bottom: 20px; right: 20px; background: #3498db; ")
    html.append("                    color: white; padding: 10px; border-radius: 5px; display: none; }")
    html.append("    hr { border: 0; border-top: 1px solid #eee; margin: 30px 0; }")
    html.append("    .description { margin: 10px 0; padding-left: 20px; border-left: 3px solid #e0e0e0; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<div class='container'>")
    
    # Title and introduction
    html.append("  <h1>RDF Schema Documentation</h1>")
    html.append("  <div class='info'>")
    html.append(f"    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>")
    html.append(f"    <strong>Source file:</strong> {os.path.basename(input_file)}<br>")
    html.append(f"    <strong>Contains:</strong> {len(classes)} classes and {len(properties)} properties</p>")
    html.append("  </div>")
    
    # Table of Contents
    html.append("  <div class='toc'>")
    html.append("    <h2 id='toc'>Table of Contents</h2>")
    html.append("    <ul>")
    html.append("      <li><a href='#namespaces'>1. Namespaces</a></li>")
    html.append("      <li><a href='#classes'>2. Classes</a>")
    html.append("        <ul>")
    for i, cls in enumerate(classes):
        cls_id = f"class-{i+1}-{cls['label'].lower().replace(' ', '-')}"
        html.append(f"          <li><a href='#{cls_id}'>{cls['label']}</a></li>")
    html.append("        </ul>")
    html.append("      </li>")
    html.append("      <li><a href='#properties'>3. Properties</a>")
    html.append("        <ul>")
    for i, prop in enumerate(properties):
        prop_id = f"property-{i+1}-{prop['label'].lower().replace(' ', '-')}"
        html.append(f"          <li><a href='#{prop_id}'>{prop['label']}</a></li>")
    html.append("        </ul>")
    html.append("      </li>")
    html.append("    </ul>")
    html.append("  </div>")
    
    # Namespaces
    html.append("  <h2 id='namespaces'>1. Namespaces</h2>")
    html.append("  <p>The following namespaces are used in this schema:</p>")
    html.append("  <table>")
    html.append("    <tr><th>Prefix</th><th>URI</th></tr>")
    for prefix, uri in sorted(namespaces.items()):
        html.append(f"    <tr><td>{prefix}</td><td class='uri'>{uri}</td></tr>")
    html.append("  </table>")
    
    # Classes
    html.append("  <h2 id='classes'>2. Classes</h2>")
    html.append(f"  <p>This schema defines {len(classes)} classes:</p>")
    
    for i, cls in enumerate(classes):
        cls_id = f"class-{i+1}-{cls['label'].lower().replace(' ', '-')}"
        html.append(f"  <h3 id='{cls_id}'>{cls['label']}</h3>")
        html.append(f"  <p><span class='uri'>{cls['uri']}</span></p>")
        
        if cls['description']:
            html.append("  <div class='description'>")
            html.append(f"    <p>{cls['description']}</p>")
            html.append("  </div>")
        
        if cls['super_classes']:
            html.append("  <p><strong>Parent classes:</strong></p>")
            html.append("  <ul>")
            for super_cls in cls['super_classes']:
                html.append(f"    <li>{super_cls['label']} <span class='uri'>{super_cls['uri']}</span></li>")
            html.append("  </ul>")
        
        if cls['sub_classes']:
            html.append("  <p><strong>Child classes:</strong></p>")
            html.append("  <ul>")
            for sub_cls in cls['sub_classes']:
                html.append(f"    <li>{sub_cls['label']} <span class='uri'>{sub_cls['uri']}</span></li>")
            html.append("  </ul>")
        
        if cls['properties']:
            html.append("  <p><strong>Properties:</strong></p>")
            html.append("  <table>")
            html.append("    <tr><th>Property</th><th>Values</th><th>Description</th></tr>")
            
            for prop in cls['properties']:
                range_str = ""
                if prop['ranges']:
                    range_labels = [r['label'] for r in prop['ranges']]
                    range_str = ", ".join(range_labels)
                
                desc = prop['description'] if prop['description'] else ""
                # Ensure HTML compatibility
                desc = desc.replace("<", "&lt;").replace(">", "&gt;")
                
                html.append(f"    <tr><td>{prop['label']}</td><td>{range_str}</td><td>{desc}</td></tr>")
            
            html.append("  </table>")
        
        html.append("  <hr>")
    
    # Properties
    html.append("  <h2 id='properties'>3. Properties</h2>")
    html.append(f"  <p>This schema defines {len(properties)} properties:</p>")
    
    for i, prop in enumerate(properties):
        prop_id = f"property-{i+1}-{prop['label'].lower().replace(' ', '-')}"
        html.append(f"  <h3 id='{prop_id}'>{prop['label']}</h3>")
        html.append(f"  <p><span class='uri'>{prop['uri']}</span><br>")
        html.append(f"  <strong>Type:</strong> {prop['type']}</p>")
        
        if prop['description']:
            html.append("  <div class='description'>")
            html.append(f"    <p>{prop['description']}</p>")
            html.append("  </div>")
        
        if prop['domains']:
            html.append("  <p><strong>Domain (classes that can have this property):</strong></p>")
            html.append("  <ul>")
            for domain in prop['domains']:
                html.append(f"    <li>{domain['label']} <span class='uri'>{domain['uri']}</span></li>")
            html.append("  </ul>")
        
        if prop['ranges']:
            html.append("  <p><strong>Range (values this property can have):</strong></p>")
            html.append("  <ul>")
            for range_info in prop['ranges']:
                html.append(f"    <li>{range_info['label']} <span class='uri'>{range_info['uri']}</span></li>")
            html.append("  </ul>")
        
        if prop['super_properties']:
            html.append("  <p><strong>Parent properties:</strong></p>")
            html.append("  <ul>")
            for super_prop in prop['super_properties']:
                html.append(f"    <li>{super_prop['label']} <span class='uri'>{super_prop['uri']}</span></li>")
            html.append("  </ul>")
        
        html.append("  <hr>")
    
    # Back to top button
    html.append("  <a href='#' id='back-to-top' class='back-to-top'>Back to Top</a>")
    
    # Footer and closing tags
    html.append("  <script>")
    html.append("    // Show/hide back to top button based on scroll position")
    html.append("    window.onscroll = function() {")
    html.append("      var backToTop = document.getElementById('back-to-top');")
    html.append("      if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {")
    html.append("        backToTop.style.display = 'block';")
    html.append("      } else {")
    html.append("        backToTop.style.display = 'none';")
    html.append("      }")
    html.append("    };")
    html.append("  </script>")
    html.append("</div>")
    html.append("</body>")
    html.append("</html>")
    
    # Join all lines into a single string
    html_text = "\n".join(html)
    
    # Save to file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_text)
        print(f"HTML documentation saved to {output_file}")
        return None
    else:
        return html_text

def generate_schema_cheatsheet(input_file, output_file=None, format='turtle'):
    """
    Generate a simplified cheat sheet for an RDF schema - just the core concepts
    without all the details, meant for quick reference.
    
    Parameters:
    input_file (str): Path to the RDF schema file
    output_file (str, optional): Path to save the cheat sheet
    format (str): Format of the RDF file (e.g., 'turtle', 'xml', 'json-ld')
    
    Returns:
    str: The generated cheat sheet text (if output_file is None)
    """
    # Load the RDF schema
    g = Graph()
    g.parse(input_file, format=format)
    
    # Create lists to store information
    classes = []
    properties = []
    
    # Extract classes
    for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
        if isinstance(s, URIRef):
            classes.append({
                'uri': str(s),
                'label': get_readable_label(s, g)
            })
    
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef) and not any(c['uri'] == str(s) for c in classes):
            classes.append({
                'uri': str(s),
                'label': get_readable_label(s, g)
            })
    
    # Extract properties
    for s, p, o in g.triples((None, RDF.type, RDF.Property)):
        if isinstance(s, URIRef):
            prop = {
                'uri': str(s),
                'label': get_readable_label(s, g),
                'domains': [],
                'ranges': []
            }
            
            # Get domains
            for _, _, domain in g.triples((s, RDFS.domain, None)):
                domain_label = get_readable_label(domain, g)
                prop['domains'].append(domain_label)
            
            # Get ranges
            for _, _, range_uri in g.triples((s, RDFS.range, None)):
                range_label = get_readable_label(range_uri, g)
                prop['ranges'].append(range_label)
                
            properties.append(prop)
    
    # Extract other property types as well
    for prop_type in [OWL.DatatypeProperty, OWL.ObjectProperty]:
        for s, p, o in g.triples((None, RDF.type, prop_type)):
            if isinstance(s, URIRef) and not any(p['uri'] == str(s) for p in properties):
                prop = {
                    'uri': str(s),
                    'label': get_readable_label(s, g),
                    'domains': [],
                    'ranges': []
                }
                
                # Get domains
                for _, _, domain in g.triples((s, RDFS.domain, None)):
                    domain_label = get_readable_label(domain, g)
                    prop['domains'].append(domain_label)
                
                # Get ranges
                for _, _, range_uri in g.triples((s, RDFS.range, None)):
                    range_label = get_readable_label(range_uri, g)
                    prop['ranges'].append(range_label)
                    
                properties.append(prop)
    
    # Sort alphabetically
    classes.sort(key=lambda x: x['label'].lower())
    properties.sort(key=lambda x: x['label'].lower())
    
    # Generate the cheat sheet
    lines = []
    
    lines.append("=" * 60)
    lines.append(" " * 15 + "RDF SCHEMA CHEAT SHEET")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"SOURCE: {os.path.basename(input_file)}")
    lines.append(f"CLASSES: {len(classes)}  |  PROPERTIES: {len(properties)}")
    lines.append("")
    lines.append("-" * 60)
    
    # Classes section
    lines.append("CLASSES")
    lines.append("=" * 7)
    lines.append("")
    
    # Figure out column width based on the longest label
    if classes:
        max_label_len = max(len(cls['label']) for cls in classes)
        col_width = max(max_label_len + 2, 30)
    else:
        col_width = 30
    
    # Format header
    lines.append(f"{'CLASS NAME':<{col_width}} URI")
    lines.append(f"{'-' * col_width} {'-' * 50}")
    
    # Add each class
    for cls in classes:
        lines.append(f"{cls['label']:<{col_width}} {cls['uri']}")
    
    lines.append("")
    lines.append("-" * 60)
    
    # Properties section
    lines.append("PROPERTIES")
    lines.append("=" * 10)
    lines.append("")
    
    # Format header for properties table
    if properties:
        max_prop_len = max(len(prop['label']) for prop in properties)
        prop_col_width = max(max_prop_len + 2, 25)
    else:
        prop_col_width = 25
    
    domain_col_width = 20
    range_col_width = 20
    
    lines.append(f"{'PROPERTY':<{prop_col_width}} {'DOMAIN':<{domain_col_width}} {'RANGE':<{range_col_width}} URI")
    lines.append(f"{'-' * prop_col_width} {'-' * domain_col_width} {'-' * range_col_width} {'-' * 30}")
    
    # Add each property
    for prop in properties:
        domains = ", ".join(prop['domains']) if prop['domains'] else "-"
        ranges = ", ".join(prop['ranges']) if prop['ranges'] else "-"
        
        # Truncate if too long
        if len(domains) > domain_col_width - 3:
            domains = domains[:domain_col_width - 5] + "..."
        if len(ranges) > range_col_width - 3:
            ranges = ranges[:range_col_width - 5] + "..."
            
        lines.append(f"{prop['label']:<{prop_col_width}} {domains:<{domain_col_width}} {ranges:<{range_col_width}} {prop['uri']}")
    
    # Join the lines and return or save
    cheatsheet = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cheatsheet)
        print(f"Cheat sheet saved to {output_file}")
        return None
    else:
        return cheatsheet

# Usage examples
# 1. Generate plain text documentation:
# text_doc = generate_schema_documentation('my_schema.ttl', 'schema_doc.txt')

# 2. Generate markdown documentation:
# md_doc = generate_schema_markdown('my_schema.ttl', 'schema_doc.md')

# 3. Generate HTML documentation:
# html_doc = generate_schema_html('my_schema.ttl', 'schema_doc.html')

# 4. Generate a simple cheat sheet:
# cheatsheet = generate_schema_cheatsheet('my_schema.ttl', 'schema_cheatsheet.txt')