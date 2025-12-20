#!/usr/bin/env python3
"""
Node Property Query Utility
Author: Pranav Singh

Simple utility for retrieving all properties of a specific node from an RDF
graph by its URI. Returns both outgoing and incoming properties.

Usage:
    python query_specific_node.py
"""

import rdflib
import json

def get_all_node_properties(rdf_file_path, target_uri):
    """
    Get ALL properties for a specific node URI
    
    Args:
        rdf_file_path: Path to RDF file
        target_uri: The exact URI of the node you want to analyze
    
    Returns:
        Complete property dictionary
    """
    print(f"Loading RDF: {rdf_file_path}")
    graph = rdflib.Graph()
    graph.parse(rdf_file_path, format='turtle')
    
    # Handle different URI types
    if target_uri.startswith('_:'):
        node = rdflib.BNode(target_uri[2:])
    else:
        node = rdflib.URIRef(target_uri)
    
    print(f"Querying node: {target_uri}")
    
    # Get all properties where this node is the SUBJECT
    outgoing = {}
    print("Getting outgoing properties (this node -> others)...")
    for predicate, obj in graph.predicate_objects(node):
        pred_str = str(predicate)
        obj_str = str(obj)
        
        if pred_str not in outgoing:
            outgoing[pred_str] = []
        outgoing[pred_str].append({
            'value': obj_str,
            'type': 'Literal' if isinstance(obj, rdflib.Literal) else 'URI',
            'datatype': str(obj.datatype) if isinstance(obj, rdflib.Literal) and obj.datatype else None,
            'language': str(obj.language) if isinstance(obj, rdflib.Literal) and obj.language else None
        })
    
    # Get all properties where this node is the OBJECT
    incoming = {}
    print("Getting incoming properties (others -> this node)...")
    for subject, predicate in graph.subject_predicates(node):
        pred_str = str(predicate)
        subj_str = str(subject)
        
        if pred_str not in incoming:
            incoming[pred_str] = []
        incoming[pred_str].append(subj_str)
    
    result = {
        'target_uri': target_uri,
        'outgoing_properties': outgoing,
        'incoming_properties': incoming,
        'stats': {
            'outgoing_property_types': len(outgoing),
            'incoming_property_types': len(incoming),
            'total_outgoing_values': sum(len(values) for values in outgoing.values()),
            'total_incoming_values': sum(len(values) for values in incoming.values())
        }
    }
    
    return result

def display_properties(properties_dict):
    """Display properties in readable format"""
    target = properties_dict['target_uri']
    outgoing = properties_dict['outgoing_properties']
    incoming = properties_dict['incoming_properties']
    stats = properties_dict['stats']
    
    print("\n" + "="*80)
    print(f"COMPLETE PROPERTIES FOR: {target}")
    print("="*80)
    
    print(f"\nSTATISTICS:")
    print(f"  Outgoing property types: {stats['outgoing_property_types']}")
    print(f"  Incoming property types: {stats['incoming_property_types']}")
    print(f"  Total outgoing values: {stats['total_outgoing_values']}")
    print(f"  Total incoming values: {stats['total_incoming_values']}")
    
    print(f"\nOUTGOING PROPERTIES (this node as subject):")
    print("-" * 60)
    for prop, values in outgoing.items():
        prop_name = prop.split('/')[-1] if '/' in prop else prop
        print(f"\n  {prop_name} ({len(values)} values):")
        print(f"    Full URI: {prop}")
        for i, val_info in enumerate(values):
            if i < 5:  # Show first 5 values
                value = val_info['value']
                val_type = val_info['type']
                if len(value) > 100:
                    value = value[:100] + "..."
                print(f"    [{val_type}] {value}")
                if val_info['datatype']:
                    print(f"         Datatype: {val_info['datatype']}")
                if val_info['language']:
                    print(f"         Language: {val_info['language']}")
            elif i == 5:
                print(f"    ... and {len(values) - 5} more values")
                break
    
    if incoming:
        print(f"\nINCOMING PROPERTIES (this node as object):")
        print("-" * 60)
        for prop, subjects in incoming.items():
            prop_name = prop.split('/')[-1] if '/' in prop else prop
            print(f"\n  {prop_name} ({len(subjects)} subjects):")
            print(f"    Full URI: {prop}")
            for i, subj in enumerate(subjects):
                if i < 3:  # Show first 3 subjects
                    print(f"    <- {subj}")
                elif i == 3:
                    print(f"    <- ... and {len(subjects) - 3} more subjects")
                    break

def interactive_node_query():
    """Interactive mode to query specific nodes"""
    rdf_file = input("Enter RDF file path (default: mappings/NKB_RDF_V3.ttl): ").strip()
    if not rdf_file:
        rdf_file = 'mappings/NKB_RDF_V3.ttl'
    
    while True:
        print("\n" + "="*50)
        print("NODE PROPERTY QUERY")
        print("="*50)
        print("Enter a node URI to analyze (or 'quit' to exit)")
        print("Examples:")
        print("  - http://example.org/material/nano-tio2-001")
        print("  - _:b123456")
        print("  - 'sample' to see sample nodes first")
        
        user_input = input("\nNode URI: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'sample':
            print("Finding sample nodes...")
            try:
                graph = rdflib.Graph()
                graph.parse(rdf_file, format='turtle')
                subjects = list(graph.subjects())[:10]
                print("Sample nodes:")
                for i, subj in enumerate(subjects):
                    print(f"  {i}: {subj}")
            except Exception as e:
                print(f"Error loading samples: {e}")
            continue
        
        if not user_input:
            print("Please enter a valid URI")
            continue
        
        try:
            properties = get_all_node_properties(rdf_file, user_input)
            display_properties(properties)
            
            # Option to save
            save = input("\nSave to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"node_properties_{user_input.replace('/', '_').replace(':', '_')}.json"
                with open(filename, 'w') as f:
                    json.dump(properties, f, indent=2)
                print(f"Saved to: {filename}")
                
        except Exception as e:
            print(f"Error analyzing node: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    interactive_node_query()
