import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

class CareerInsights:
    def __init__(self, data_path='recommender-data/processed/processed_dataset.csv'):
        """Initialize the insights module with data path"""
        self.data_path = data_path
        self.df = None
        
        # Create directories for outputs
        os.makedirs('recommender-insights/plots', exist_ok=True)
        os.makedirs('recommender-insights/reports', exist_ok=True)
        
    def load_data(self):
        """Load data from the processed dataset"""
        print("Loading data from processed dataset...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Successfully loaded data from {self.data_path}")
            print(f"Dataset shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def visualize_career_distribution(self, pdf):
        """Visualize the distribution of career aspirations"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
        
        print("Generating career distribution visualization...")
        
        # Get the target column name
        target_col = 'career_aspiration' if 'career_aspiration' in self.df.columns else 'Career_Field'
        
        # Create a new figure for PDF
        plt.figure(figsize=(10, 6))
        
        career_counts = self.df[target_col].value_counts()
        
        # Plot only top 15 careers if there are many
        if len(career_counts) > 15:
            career_counts = career_counts.head(15)
            title_suffix = " (Top 15)"
        else:
            title_suffix = ""
            
        sns.barplot(x=career_counts.values, y=career_counts.index)
        plt.title(f'Distribution of Career Aspirations{title_suffix}')
        plt.xlabel('Number of Students')
        plt.tight_layout()
        
        # Save to PDF
        pdf.savefig()
        plt.close()
        
        # Also save as PNG for reference
        plt.figure(figsize=(12, 6))
        sns.barplot(x=career_counts.values, y=career_counts.index)
        plt.title(f'Distribution of Career Aspirations{title_suffix}')
        plt.xlabel('Number of Students')
        plt.tight_layout()
        plt.savefig('recommender-insights/plots/career_distribution.png')
        plt.close()
        
        # Add text information to PDF
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        total = len(self.df)
        text = f"Career Distribution:\n\n"
        for career, count in career_counts.items():
            percentage = (count / total) * 100
            text += f"• {career}: {count} students ({percentage:.1f}%)\n"
        plt.text(0.1, 0.9, text, va='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    
    def visualize_correlation_matrix(self, pdf):
        """Visualize the correlation matrix between subject scores"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
        
        print("Generating correlation matrix visualization...")
        
        # Find subject score columns
        subject_cols = [col for col in self.df.columns if '_score' in col]
        
        if not subject_cols:
            print("No subject score columns found in the dataset.")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[subject_cols].corr()
        
        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                   xticklabels=[col.replace('_score', '').capitalize() for col in subject_cols],
                   yticklabels=[col.replace('_score', '').capitalize() for col in subject_cols])
        plt.title('Correlation Matrix of Subject Scores')
        plt.tight_layout()
        
        # Save to PDF
        pdf.savefig()
        plt.close()
        
        # Also save as PNG for reference
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                   xticklabels=[col.replace('_score', '').capitalize() for col in subject_cols],
                   yticklabels=[col.replace('_score', '').capitalize() for col in subject_cols])
        plt.title('Correlation Matrix of Subject Scores')
        plt.tight_layout()
        plt.savefig('recommender-insights/plots/correlation_matrix.png')
        plt.close()
        
        # Add text information to PDF
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        text = "Strongest Subject Correlations:\n\n"
        
        corr_pairs = []
        for i, subject1 in enumerate(subject_cols):
            for subject2 in subject_cols[i+1:]:
                corr_value = corr_matrix.loc[subject1, subject2]
                corr_pairs.append((subject1, subject2, corr_value))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for subject1, subject2, corr_value in corr_pairs[:5]:
            subject1_name = subject1.replace('_score', '').capitalize()
            subject2_name = subject2.replace('_score', '').capitalize()
            text += f"• {subject1_name} & {subject2_name}: {corr_value:.3f}\n"
        
        plt.text(0.1, 0.9, text, va='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    
    def visualize_career_subject_relationship(self, pdf):
        """Visualize the relationship between careers and subject scores"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
        
        print("Generating career-subject relationship visualization...")
        
        # Get the target column name
        target_col = 'career_aspiration' if 'career_aspiration' in self.df.columns else 'Career_Field'
        
        # Find subject score columns
        subject_cols = [col for col in self.df.columns if '_score' in col]
        
        if not subject_cols:
            print("No subject score columns found in the dataset.")
            return
        
        # Get top careers by frequency
        top_careers = self.df[target_col].value_counts().head(8).index
        
        # Calculate average scores by career
        career_avg_scores = self.df[self.df[target_col].isin(top_careers)].groupby(target_col)[subject_cols].mean()
        
        # Create a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(career_avg_scores, annot=True, cmap='YlGnBu', fmt='.1f',
                   xticklabels=[col.replace('_score', '').capitalize() for col in subject_cols])
        plt.title('Average Subject Scores by Career Field')
        plt.tight_layout()
        
        # Save to PDF
        pdf.savefig()
        plt.close()
        
        # Also save as PNG for reference
        plt.figure(figsize=(12, 8))
        sns.heatmap(career_avg_scores, annot=True, cmap='YlGnBu', fmt='.1f',
                   xticklabels=[col.replace('_score', '').capitalize() for col in subject_cols])
        plt.title('Average Subject Scores by Career Field')
        plt.tight_layout()
        plt.savefig('recommender-insights/plots/career_subject_heatmap.png')
        plt.close()
        
        # Add text information to PDF
        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        text = "Top Subjects by Career:\n\n"
        
        for career in top_careers:
            if career in career_avg_scores.index:
                top_subjects = career_avg_scores.loc[career].sort_values(ascending=False).head(3)
                text += f"{career}:\n"
                for subject, score in top_subjects.items():
                    subject_name = subject.replace('_score', '').capitalize()
                    text += f"• {subject_name}: {score:.1f}/100\n"
                text += "\n"
        
        plt.text(0.1, 0.95, text, va='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    
    def add_dataset_summary(self, pdf):
        """Add dataset summary to the PDF"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
            
        fig = plt.figure(figsize=(10, 8))
        plt.axis('off')
        
        # Get the target column name
        target_col = 'career_aspiration' if 'career_aspiration' in self.df.columns else 'Career_Field'
        
        # Find subject score columns
        subject_cols = [col for col in self.df.columns if '_score' in col]
        
        text = "CAREER RECOMMENDATION SYSTEM\nData Insights Report\n\n"
        text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        text += "Dataset Summary:\n\n"
        text += f"• Total records: {self.df.shape[0]}\n"
        text += f"• Number of unique careers: {self.df[target_col].nunique()}\n"
        text += f"• Subject areas analyzed: {len(subject_cols)}\n\n"
        
        text += "Subject Score Statistics:\n\n"
        for subject in subject_cols:
            subject_name = subject.replace('_score', '').capitalize()
            mean = self.df[subject].mean()
            median = self.df[subject].median()
            std = self.df[subject].std()
            text += f"• {subject_name}: Mean={mean:.1f}, Median={median:.1f}, Std={std:.1f}\n"
        
        plt.text(0.1, 0.95, text, va='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    
    def add_key_insights(self, pdf):
        """Add key insights page to the PDF"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
            
        fig = plt.figure(figsize=(10, 8))
        plt.axis('off')
        
        # Get the target column name
        target_col = 'career_aspiration' if 'career_aspiration' in self.df.columns else 'Career_Field'
        
        # Find subject score columns
        subject_cols = [col for col in self.df.columns if '_score' in col]
        
        # Calculate correlation matrix
        corr_matrix = self.df[subject_cols].corr()
        
        # Get strongest correlations
        corr_pairs = []
        for i, subject1 in enumerate(subject_cols):
            for subject2 in subject_cols[i+1:]:
                corr_value = corr_matrix.loc[subject1, subject2]
                corr_pairs.append((subject1, subject2, corr_value))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Get top careers
        top_careers = self.df[target_col].value_counts().head(5)
        
        text = "Key Insights:\n\n"
        
        # Career distribution insights
        text += "1. Career Distribution:\n"
        text += f"• The most popular career aspiration is {top_careers.index[0]} with {top_careers.values[0]} students ({(top_careers.values[0]/len(self.df))*100:.1f}% of total).\n"
        text += f"• The top 5 careers account for {(top_careers.sum()/len(self.df))*100:.1f}% of all student aspirations.\n\n"
        
        # Subject correlation insights
        text += "2. Subject Correlations:\n"
        if len(corr_pairs) > 0:
            strongest = corr_pairs[0]
            subject1_name = strongest[0].replace('_score', '').capitalize()
            subject2_name = strongest[1].replace('_score', '').capitalize()
            text += f"• The strongest correlation is between {subject1_name} and {subject2_name} ({strongest[2]:.2f}).\n"
            text += "• Most subject correlations are relatively weak, suggesting that students tend to have distinct strengths rather than performing uniformly across all subjects.\n\n"
        
        # Career-specific insights
        text += "3. Career-Specific Insights:\n"
        
        # Get average scores by career for all careers
        career_avg_scores = self.df.groupby(target_col)[subject_cols].mean()
        
        # For Software Engineers
        if "Software Engineer" in career_avg_scores.index:
            se_scores = career_avg_scores.loc["Software Engineer"]
            top_se_subject = se_scores.idxmax().replace('_score', '').capitalize()
            text += f"• Software Engineers excel in {top_se_subject} with an average score of {se_scores.max():.1f}.\n"
        
        # For Doctors
        if "Doctor" in career_avg_scores.index:
            doc_scores = career_avg_scores.loc["Doctor"]
            top_doc_subject = doc_scores.idxmax().replace('_score', '').capitalize()
            text += f"• Students aspiring to be Doctors show strongest performance in {top_doc_subject}.\n"
        
        # For Lawyers
        if "Lawyer" in career_avg_scores.index:
            law_scores = career_avg_scores.loc["Lawyer"]
            top_law_subject = law_scores.idxmax().replace('_score', '').capitalize()
            text += f"• Aspiring Lawyers demonstrate highest proficiency in {top_law_subject}.\n\n"
        
        text += "4. Recommendations:\n"
        text += "• Career guidance should focus on subject strengths, as there are clear patterns in the subject preferences of students aspiring to different careers.\n"
        text += "• The recommendation system should consider the unique subject score profiles for each career path when making suggestions.\n"
        text += "• Students with balanced scores across multiple subjects may have more career flexibility and could benefit from exploring multiple career options."
        
        plt.text(0.1, 0.95, text, va='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    
    def generate_pdf_report(self):
        """Generate a comprehensive PDF report with all visualizations and insights"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
            
        pdf_path = 'recommender-insights/reports/career_insights_report.pdf'
        print(f"Generating PDF report at {pdf_path}...")
        
        with PdfPages(pdf_path) as pdf:
            # Add title page and dataset summary
            self.add_dataset_summary(pdf)
            
            # Add visualizations
            self.visualize_career_distribution(pdf)
            self.visualize_correlation_matrix(pdf)
            self.visualize_career_subject_relationship(pdf)
            
            # Add key insights page
            self.add_key_insights(pdf)
        
        print(f"✅ PDF report generated successfully at {pdf_path}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        self.load_data()
        if self.df is not None:
            self.generate_pdf_report()
            print("\n✅ Analysis complete!")
        else:
            print("❌ Could not generate report due to data loading error.")


if __name__ == "__main__":
    insights = CareerInsights()
    insights.run_analysis()