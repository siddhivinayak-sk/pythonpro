"""
Using official CycloneDX library for dependency analysis
"""
from cyclonedx.model.bom import Bom
import json
from typing import Set, List, Dict, Optional, Tuple
import sys
import boto3
import argparse

class CycloneDXDependencyAnalyzer:
    """Analyze dependencies using official CycloneDX library"""
    
    def __init__(self, bom_json_path: str = None, bom_json_data: dict = None):
        """Initialize with BOM JSON file path or JSON data"""
        if bom_json_data:
            self.bom_data = bom_json_data
        elif bom_json_path:
            with open(bom_json_path, 'r') as f:
                self.bom_data = json.load(f)
        else:
            raise ValueError("Either bom_json_path or bom_json_data must be provided")
        
        # Create BOM object and populate from JSON
        self.bom = Bom()
        self._parse_components(self.bom_data)
        self._parse_dependencies(self.bom_data)
    
    def _parse_components(self, bom_data: dict):
        """Parse components from raw JSON data"""
        from cyclonedx.model.component import Component
        from cyclonedx.model import XsUri
        
        if 'components' in bom_data:
            for comp_data in bom_data['components']:
                component = Component(
                    name=comp_data.get('name', ''),
                    version=comp_data.get('version'),
                    bom_ref=XsUri(comp_data.get('bom-ref', ''))
                )
                self.bom.components.add(component)
    
    def _parse_dependencies(self, bom_data: dict):
        """Parse dependencies section to build dependency map"""
        self.dependency_map = {}
        if 'dependencies' in bom_data:
            for dep in bom_data['dependencies']:
                ref = dep.get('ref', '')
                depends_on = dep.get('dependsOn', [])
                self.dependency_map[ref] = depends_on
    
    def get_direct_dependencies(self, component_ref: str) -> Set[str]:
        """Get direct dependencies for a component"""
        return set(self.dependency_map.get(component_ref, []))
    
    def get_all_dependencies_recursive(self, component_ref: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """Get all dependencies (direct + transitive) recursively"""
        if visited is None:
            visited = set()
        
        if component_ref in visited:
            return set()
        
        visited.add(component_ref)
        all_deps = set()
        
        # Get direct dependencies
        direct = self.dependency_map.get(component_ref, [])
        all_deps.update(direct)
        
        # Recursively get transitive dependencies
        for dep in direct:
            transitive = self.get_all_dependencies_recursive(dep, visited)
            all_deps.update(transitive)
        
        return all_deps
    
    def identify_transitive_dependencies(self, component_ref: str) -> Tuple[Set[str], Set[str]]:
        """
        Identify direct vs transitive dependencies for a component
        Returns: (direct_dependencies, transitive_dependencies)
        """
        direct_deps = self.get_direct_dependencies(component_ref)
        all_deps = self.get_all_dependencies_recursive(component_ref)
        
        # Transitive dependencies = all dependencies - direct dependencies
        transitive_deps = all_deps - direct_deps
        
        return direct_deps, transitive_deps
    
    def get_component_name(self, bom_ref: str) -> str:
        """Get component name from bom-ref"""
        for comp_data in self.bom_data.get('components', []):
            if comp_data.get('bom-ref') == bom_ref:
                name = comp_data.get('name', 'Unknown')
                version = comp_data.get('version', '')
                return f"{name}@{version}" if version else name
        return bom_ref
    
    def find_component_by_name(self, name: str, version: str = None) -> List[Dict[str, Optional[str]]]:
        """
        Find components by name or purl/bom-ref pattern, with optional version filter.

        Supports inputs like:
          - 'spring-web'                        (simple artifact name)
          - 'org.springframework/spring-web'    (group/artifact as it appears in purl)
        When version is supplied the search matches the combined
        '<name>@<version>' substring inside the purl / bom-ref fields so that
        e.g. 'org.springframework/spring-web' + '6.1.16' resolves to the entry
        whose purl contains 'org.springframework/spring-web@6.1.16'.
        """
        results = []
        name_lower = name.lower()
        # Build the combined pattern used for purl / bom-ref matching
        name_version_pattern = f"{name_lower}@{version}".lower() if version else None

        for comp_data in self.bom_data.get('components', []):
            comp_name    = comp_data.get('name', '').lower()
            comp_version = comp_data.get('version', '')
            comp_purl    = comp_data.get('purl', '').lower()
            comp_bom_ref = comp_data.get('bom-ref', '').lower()

            if name_version_pattern:
                # Exact group/artifact@version match in purl or bom-ref
                name_matches = (
                    name_version_pattern in comp_purl or
                    name_version_pattern in comp_bom_ref
                )
            else:
                # Fallback: substring match in name, purl, or bom-ref
                name_matches = (
                    name_lower in comp_name or
                    name_lower in comp_purl or
                    name_lower in comp_bom_ref
                )

            if not name_matches:
                continue

            # Additional version guard when no combined pattern was used
            if version and not name_version_pattern and comp_version != version:
                continue

            results.append({
                'bom_ref': comp_data.get('bom-ref', ''),
                'name': comp_data.get('name', ''),
                'version': comp_version if comp_version else None
            })

        return results
    
    def find_transitive_for_library(self, library_name: str, library_version: str = None) -> Dict[str, any]:
        """
        Find transitive dependencies for a specific library by name (and optional version).
        Returns detailed information about the library and its dependencies.
        """
        results = self.find_component_by_name(library_name, library_version)

        if not results:
            label = f"'{library_name}@{library_version}'" if library_version else f"'{library_name}'"
            return {
                'found': False,
                'library_name': library_name,
                'library_version': library_version,
                'message': f"No component found matching {label}"
            }
        
        # If multiple matches, analyze all of them
        all_matches = []
        for result in results:
            bom_ref = result['bom_ref']
            direct_deps, transitive_deps = self.identify_transitive_dependencies(bom_ref)
            
            # Get component names for dependencies
            direct_deps_named = [
                {
                    'bom_ref': dep,
                    'name': self.get_component_name(dep)
                } for dep in direct_deps
            ]
            
            transitive_deps_named = [
                {
                    'bom_ref': dep,
                    'name': self.get_component_name(dep)
                } for dep in transitive_deps
            ]
            
            all_matches.append({
                'component': result,
                'direct_count': len(direct_deps),
                'transitive_count': len(transitive_deps),
                'direct_dependencies': direct_deps_named,
                'transitive_dependencies': transitive_deps_named
            })
        
        return {
            'found': True,
            'library_name': library_name,
            'matches': all_matches
        }


def fetch_sbom_from_s3(bucket: str, key: str, region: str = None,
                       access_key: str = None, secret_key: str = None) -> dict:
    """
    Fetch SBOM JSON from AWS S3
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path to the SBOM JSON file)
        region: AWS region (optional)
        access_key: AWS access key ID (optional, falls back to default credential chain)
        secret_key: AWS secret access key (optional, falls back to default credential chain)
    
    Returns:
        dict: SBOM JSON data
    """
    try:
        client_kwargs = {}
        if region:
            client_kwargs['region_name'] = region
        if access_key and secret_key:
            client_kwargs['aws_access_key_id'] = access_key
            client_kwargs['aws_secret_access_key'] = secret_key
        s3_client = boto3.client('s3', **client_kwargs)
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        sbom_data = json.loads(response['Body'].read().decode('utf-8'))
        return sbom_data
    except Exception as e:
        raise Exception(f"Error fetching SBOM from S3: {str(e)}")


def analyze_infected_package(component_name: str, component_version: str,product_name: str, product_version: str, 
                            aws_bucket_name: str = None, aws_s3_prefix: str = "", aws_client_region: str = None,
                            aws_client_id: str = None, secret_key: str = None) -> dict:
    """
    Analyze SBOM for an infected package and return main + transitive dependencies        
    
    Returns:
        dict: JSON output with main and transitive dependency information
    """
    # Build SBOM filename
    sbom_filename = f"sbom_{component_name}-{component_version}.json"
    
    try:
        if not aws_bucket_name:
            raise ValueError("bucket must be provided")
        
        if aws_s3_prefix:
            s3_key = f"{aws_s3_prefix.rstrip('/')}/{sbom_filename}"
        else:
            s3_key = sbom_filename
        
        sbom_data = fetch_sbom_from_s3(aws_bucket_name, s3_key, aws_client_region, aws_client_id, secret_key)
        location = f"s3://{aws_bucket_name}/{s3_key}"
        
        # Initialize analyzer
        analyzer = CycloneDXDependencyAnalyzer(bom_json_data=sbom_data)
        
        # Find the infected package by group/artifact name and version
        # e.g. product_name='org.springframework/spring-web', product_version='6.1.16'
        # resolves against purl entries like 'pkg:maven/org.springframework/spring-web@6.1.16'
        result = analyzer.find_transitive_for_library(product_name, product_version)

        if not result['found']:
            return {
                'status': 'not_found',
                'component_name': component_name,
                'component_version': component_version,
                'product_name': product_name,
                'product_version': product_version,
                'message': result['message']
            }
        
        # Build output JSON
        output = {
            'status': 'success',
            'component_name': component_name,
            'component_version': component_version,
            'product_name': product_name,
            'product_version': product_version,
            'sbom_filename': sbom_filename,
            'location': location,
            'matches': []
        }
        
        # Group matches by version to consolidate results
        version_groups = {}
        for match in result['matches']:
            comp_version = match['component']['version']
            if comp_version not in version_groups:
                version_groups[comp_version] = {
                    'component_names': set(),
                    'bom_refs': [],
                    'direct_deps': set(),
                    'transitive_deps': set()
                }
            
            version_groups[comp_version]['component_names'].add(match['component']['name'])
            version_groups[comp_version]['bom_refs'].append(match['component']['bom_ref'])
            version_groups[comp_version]['direct_deps'].update([dep['name'] for dep in match['direct_dependencies']])
            version_groups[comp_version]['transitive_deps'].update([dep['name'] for dep in match['transitive_dependencies']])
        
        # Process each version group
        for comp_version, group_data in version_groups.items():
            match_output = {
                'component': {
                    'name': list(group_data['component_names'])[0] if len(group_data['component_names']) == 1 else ', '.join(sorted(group_data['component_names'])),
                    'version': comp_version,
                    'bom_refs': group_data['bom_refs']
                },
                'main_dependencies': {
                    'count': len(group_data['direct_deps']),
                    'dependencies': sorted(list(group_data['direct_deps']))
                },
                'transitive_dependencies': {
                    'count': len(group_data['transitive_deps']),
                    'dependencies': sorted(list(group_data['transitive_deps']))
                }
            }
            output['matches'].append(match_output)
        
        return output
    
    except Exception as e:
        return {
            'status': 'error',
            'component_name': component_name,
            'component_version': component_version,
            'product_name': product_name,
            'product_version': product_version,
            'error': str(e)
        }


# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CycloneDX SBOM Analyzer - Analyze dependencies and find transitive dependencies',
        formatter_class=argparse.RawDescriptionHelpFormatter,        
    )
    
    # Input mode selection
    parser.add_argument('--component-name', type=str, help='Component name (for infected package analysis)')
    parser.add_argument('--component-version', type=str, help='Component version (for infected package analysis)')
    parser.add_argument('--product-name', type=str, help='Infected product/package name to find (for infected package analysis)')
    parser.add_argument('--product-version', type=str, help='Infected product/package version to find (for infected package analysis)')
    parser.add_argument('--aws-s3-prefix', type=str, default="", help='S3 prefix/path for SBOM files (optional)')
    
    parser.add_argument('-b', '--aws-bucket-name', type=str, help='AWS S3 bucket name')
    parser.add_argument('--aws-client-region', type=str, help='AWS region for S3 (optional)')
    parser.add_argument('--aws-client-id', type=str, help='AWS access key ID (optional, falls back to default credential chain)')
    parser.add_argument('--aws-client-secret', type=str, help='AWS secret access key (optional, falls back to default credential chain)')
    
    args = parser.parse_args()
    
    # Check for infected package analysis mode
    if args.component_name or args.component_version or args.product_name:
        # Validate all required arguments for infected package mode
        if not all([args.component_name, args.component_version, args.product_name, args.product_version]):
            parser.error("--component-name, --component-version, --product-name, and --product-version are all required for infected package analysis")
        
        # Validate that bucket is provided
        if not args.aws_bucket_name:
            parser.error("--aws-bucket-name must be provided for infected package analysis")

        # Run infected package analysis
        result = analyze_infected_package(
            component_name=args.component_name,
            component_version=args.component_version,
            product_name=args.product_name,
            product_version=args.product_name,
            aws_bucket_name=args.aws_bucket_name,
            aws_s3_prefix=args.aws_s3_prefix,
            aws_client_region=args.aws_client_region,
            aws_client_id=args.aws_client_id,
            aws_client_secret=args.aws_client_secret
        )
        
        # Output as JSON
        print(json.dumps(result, indent=2))
        sys.exit(0 if result['status'] == 'success' else 1)
    
    parser.error("This tool requires --component-name, --component-version, --product-name and --product-version for infected package analysis mode") 