import os
import sys
import argparse
from datetime import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi

def check_mongodb_storage(output_file=None):
    # Get MongoDB connection string from environment variable
    connection_string = os.environ.get('MONGO_URI')
    
    if not connection_string:
        print("Error: Missing MongoDB connection string (MONGO_URI)")
        sys.exit(1)
    
    try:
        # Connect to MongoDB using the same pattern as your app
        client = MongoClient(connection_string, server_api=ServerApi("1"))
        
        # Connect to your specific database
        db = client["infact_db_v3"]
        
        # Get list of all databases in the cluster
        all_dbs = client.list_database_names()
        
        # MongoDB Atlas free tier limit (512MB)
        free_tier_limit_mb = 512
        
        # Track storage metrics
        total_size_bytes = 0
        database_sizes = {}
        collection_sizes = {}
        
        # Collect size information for each database
        for db_name in all_dbs:
            if db_name not in ['admin', 'local', 'config']:  # Skip system databases
                current_db = client[db_name]
                db_stats = current_db.command('dbStats')
                db_size_bytes = db_stats.get('storageSize', 0)
                total_size_bytes += db_size_bytes
                database_sizes[db_name] = db_size_bytes / (1024 * 1024)  # Convert to MB
                
                # Get collection sizes within each database
                collection_sizes[db_name] = {}
                for collection_name in current_db.list_collection_names():
                    coll_stats = current_db.command('collStats', collection_name)
                    coll_size_mb = coll_stats.get('storageSize', 0) / (1024 * 1024)
                    collection_sizes[db_name][collection_name] = coll_size_mb
        
        # Calculate storage metrics
        total_size_mb = total_size_bytes / (1024 * 1024)
        usage_percentage = (total_size_mb / free_tier_limit_mb) * 100
        remaining_mb = free_tier_limit_mb - total_size_mb
        
        # Prepare the report content
        report = []
        report.append("# MongoDB Atlas Storage Usage Report")
        report.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append("")
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total storage used:** {total_size_mb:.2f} MB")
        report.append(f"- **Free tier limit:** {free_tier_limit_mb:.2f} MB")
        report.append(f"- **Remaining storage:** {remaining_mb:.2f} MB")
        report.append(f"- **Usage percentage:** {usage_percentage:.2f}%")
        
        # Add status indicator
        if usage_percentage > 90:
            report.append(f"- **Status:** 🔴 **CRITICAL** - {usage_percentage:.2f}% of storage limit used")
        elif usage_percentage > 75:
            report.append(f"- **Status:** 🟠 **WARNING** - {usage_percentage:.2f}% of storage limit used")
        elif usage_percentage > 50:
            report.append(f"- **Status:** 🟡 **CAUTION** - {usage_percentage:.2f}% of storage limit used")
        else:
            report.append(f"- **Status:** 🟢 **GOOD** - {usage_percentage:.2f}% of storage limit used")
        
        report.append("")
        report.append("## Database Breakdown")
        report.append("")
        report.append("| Database | Size (MB) | Percentage |")
        report.append("|----------|-----------|------------|")
        
        for db_name, size_mb in sorted(database_sizes.items(), key=lambda x: x[1], reverse=True):
            percentage = (size_mb/total_size_mb)*100 if total_size_mb > 0 else 0
            report.append(f"| {db_name} | {size_mb:.2f} | {percentage:.2f}% |")
        
        report.append("")
        report.append("## Collection Details")
        report.append("")
        
        for db_name, collections in sorted(collection_sizes.items()):
            if database_sizes.get(db_name, 0) > 1.0:  # Only show databases > 1MB
                report.append(f"### {db_name}")
                report.append("")
                report.append("| Collection | Size (MB) |")
                report.append("|------------|-----------|")
                
                for coll_name, coll_size_mb in sorted(collections.items(), key=lambda x: x[1], reverse=True):
                    if coll_size_mb > 0.1:  # Only show collections > 0.1MB
                        report.append(f"| {coll_name} | {coll_size_mb:.2f} |")
                
                report.append("")
        
        report.append("## Statistics")
        report.append("")
        report.append(f"- **Total databases:** {len(database_sizes)}")
        report.append(f"- **Total collections:** {sum(len(colls) for colls in collection_sizes.values())}")
        report.append("")
        
        # Generate full report as a string
        report_text = "\n".join(report)
        
        # Print to console
        print(report_text)
        
        # Save to file if specified
        if output_file:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")
            
        # Return exit code based on threshold
        if usage_percentage > 50:
            sys.exit(1)  # Exit with error code for GitHub Actions to detect
            
    except Exception as e:
        error_message = f"Error checking MongoDB storage: {str(e)}"
        print(error_message)
        
        if output_file:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(f"# MongoDB Storage Check Error\n\n{error_message}\n\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check MongoDB Atlas storage usage')
    parser.add_argument('--output-file', type=str, help='File to save the report to')
    args = parser.parse_args()
    
    check_mongodb_storage(args.output_file)